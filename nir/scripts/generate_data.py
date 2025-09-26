import os
import time
import random
import json
import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from owlapy import owl_expression_to_dl
from owlapy.class_expression import (
    OWLObjectUnionOf,
    OWLObjectIntersectionOf,
    OWLObjectSomeValuesFrom,
    OWLObjectAllValuesFrom,
    OWLObjectMinCardinality,
    OWLObjectMaxCardinality,
    OWLObjectOneOf,
)
from owlapy.render import DLSyntaxObjectRenderer
from owlapy.parser import DLSyntaxParser
from owlapy.utils import simplify_class_expression
from owlapy import dl_to_owl_expression
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.utils import concept_reducer, concept_reducer_properties
from ontolearn.utils.static_funcs import concept_len

from sklearn.model_selection import train_test_split


"""
How to run? E.g.,

python nir/scripts/generate_data.py --dataset_dir datasets/semantic_bible/
"""

def execute(args):
    dataset_dir = args.dataset_dir
    print(f"\n=================> Generating data on {dataset_dir} <======================\n")
    # Fix the random seed.
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    kb = KnowledgeBase(path=args.dataset_dir+"/kb/ontology.owl")
    namespace = list(kb.ontology.classes_in_signature())[0].iri.get_namespace()
    dls_renderer = DLSyntaxObjectRenderer()
    
    # R: Extract object properties.
    object_properties = {i for i in kb.get_object_properties()}
    ratio_sample_object_prop = args.ratio_sample_object_prop
    ratio_sample_nc = args.ratio_sample_nc
    # Subsample if required.
    if ratio_sample_object_prop:
        object_properties = {i for i in random.sample(population=list(object_properties),
                                                      k=max(1, int(len(
                                                          object_properties) * ratio_sample_object_prop)))}
    # R⁻: Inverse of object properties.
    object_properties_inverse = {i.get_inverse_property() for i in object_properties}
    # (5) R*: R UNION R⁻.
    object_properties_and_inverse = object_properties.union(object_properties_inverse)
    # NC: Named owl concepts.
    nc = {i for i in kb.get_concepts()}
    if ratio_sample_nc:
        # Subsample if required.
        nc = {i for i in
              random.sample(population=list(nc), k=max(1, int(len(nc) * ratio_sample_nc)))}

    # NC⁻: Complement of NC.
    nnc = {i.get_object_complement_of() for i in nc}
    # UNNC: NC UNION NC⁻.
    unnc = nc.union(nnc)
    # NC UNION NC.
    unions = concept_reducer(nc, opt=OWLObjectUnionOf)
    # NC INTERSECTION NC.
    intersections = concept_reducer(nc, opt=OWLObjectIntersectionOf)
    # NC INTERSECTION UNNC.
    intersections_nc_nnc = set()
    for _ in range(2):
        intersections_nc_nnc.update(set(map(OWLObjectIntersectionOf, zip(random.sample(list(nc),len(nc)), random.sample(list(nnc),min(20, len(nnc)))))))
    #  NC UNION UNNC.
    unions_nc_nnc = set()
    for _ in range(2):
        unions_nc_nnc.update(set(map(OWLObjectUnionOf, zip(random.sample(list(nc),len(nc)), random.sample(list(nnc),min(20, len(nnc)))))))
    # \exist r. C s.t. C \in NC and r \in R* .
    exist_nc = concept_reducer_properties(
        concepts=nc,
        properties=object_properties_and_inverse,
        cls=OWLObjectSomeValuesFrom,
    )
    # \forall r. C s.t. C \in NC and r \in R* .
    for_all_nc = concept_reducer_properties(
        concepts=nc,
        properties=object_properties_and_inverse,
        cls=OWLObjectAllValuesFrom,
    )
    # >= n r. C  and =< n r. C
    min_cardinality_nc_1, min_cardinality_nc_2, min_cardinality_nc_3 = (
        concept_reducer_properties(
            concepts=nc,
            properties=object_properties_and_inverse,
            cls=OWLObjectMinCardinality,
            cardinality=i,
        )
        for i in [1, 2, 3]
    )
    max_cardinality_nc_1, max_cardinality_nc_2, max_cardinality_nc_3 = (
        concept_reducer_properties(
            concepts=nc,
            properties=object_properties_and_inverse,
            cls=OWLObjectMaxCardinality,
            cardinality=i,
        )
        for i in [1, 2, 3]
    )
    
    # Unions and intersections of existential and universal role restrictions
    long_concepts = set()
    for _ in range(5):
        long_concepts.update(set(map(OWLObjectIntersectionOf, zip(random.sample(list(exist_nc),len(exist_nc)), random.sample(list(exist_nc),len(exist_nc))))))
        long_concepts.update(set(map(OWLObjectUnionOf, zip(random.sample(list(exist_nc),len(exist_nc)), random.sample(list(exist_nc),len(exist_nc))))))
        long_concepts.update(set(map(OWLObjectIntersectionOf, zip(random.sample(list(for_all_nc),len(for_all_nc)), random.sample(list(for_all_nc),len(for_all_nc))))))
        long_concepts.update(set(map(OWLObjectUnionOf, zip(random.sample(list(for_all_nc),len(for_all_nc)), random.sample(list(for_all_nc),len(for_all_nc))))))
    
    
    concepts_list = []
    for concept_set in tqdm([unions,
            intersections, nnc, exist_nc, for_all_nc, unions_nc_nnc, intersections_nc_nnc]):
        if len(concepts_list) >= args.max_concepts:
            break
        for concept in list(concept_set):
            try:
                simple_concept = simplify_class_expression(concept)
            except Exception as e:
                simple_concept = concept
                print("Failed on ", dls_renderer.render(concept))
            if kb.individuals_count(simple_concept):
                concepts_list.append(simple_concept)
                
    card_res = min_cardinality_nc_1.union(min_cardinality_nc_2).union(min_cardinality_nc_3).union(max_cardinality_nc_1).union(max_cardinality_nc_2).union(max_cardinality_nc_3)
    for concept in card_res:
        if len(concepts_list) >= args.max_concepts:
            break
        try:
            simple_concept = simplify_class_expression(concept)
        except Exception as e:
            simple_concept = concept
            print("Failed on ", dls_renderer.render(concept))
        if kb.individuals_count(simple_concept):
            concepts_list.append(simple_concept)
    
    if len(concepts_list) < args.max_concepts:
        print("\nAdding long concepts...")
        for concept in list(long_concepts):
            try:
                simple_concept = simplify_class_expression(concept)
            except Exception as e:
                simple_concept = concept
                print("Failed on ", dls_renderer.render(concept))
            if kb.individuals_count(simple_concept):
                concepts_list.append(simple_concept)
            if len(concepts_list) >= args.max_concepts:
                print("***Reached maximum number of concepts.\n")
                break
    concepts_list_str = [dls_renderer.render(concept) for concept in concepts_list]
    print("\n")
    print(f"Total generated concepts: {len(concepts_list_str)}.")
    concept_type = [type(concept).__name__ for concept in concepts_list]
    train_concepts, test_concepts = train_test_split(concepts_list_str, test_size=0.2, stratify=concept_type)

    if not os.path.exists(args.dataset_dir.rstrip("/")+"/data"):
        os.makedirs(args.dataset_dir.rstrip("/")+"/data")

    with open(args.dataset_dir.rstrip("/")+"/data/train_data.json", "w") as f:
        json.dump(train_concepts, f)

    with open(args.dataset_dir.rstrip("/")+"/data/test_data.json", "w") as f:
        json.dump(test_concepts, f)

    print("\nDone.")
    
def get_default_arguments(description=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="The path of a folder containing training_data.json, which contains a list of class expressions or list of tuples where the first elements are class expressions."
                             ",e.g., datasets/carcinogenesis/")
    parser.add_argument("--ratio_sample_object_prop", type=float, default=None,
                        help="To sample OWL Object Properties.")
    parser.add_argument("--ratio_sample_nc", type=float, default=None,
                        help="To sample OWL Classes.")
    parser.add_argument("--max_concepts", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = get_default_arguments()
    execute(args)


if __name__ == "__main__":
    main()
