import os
import time
import random
import argparse
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoConfig

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

from ontolearn.utils import (
    jaccard_similarity,
    f1_set_similarity,
    concept_reducer,
    concept_reducer_properties,
)

from ontolearn.triple_store import TripleStore
from ontolearn.knowledge_base import KnowledgeBase

"""
How to run? E.g.,

python tentris_alc_retrieval_eval.py --dataset_dir datasets/semantic_bible/ --output_dir Retrieval_semb_tentris
"""

def execute(args):
    dataset_dir = args.dataset_dir
    print(f"\n=================> Running TentrisALC on {dataset_dir} <======================\n")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # Fix the random seed.
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    kb = KnowledgeBase(path=args.dataset_dir+"/kb/ontology.owl")
    tentris_kb = TripleStore(url="http://0.0.0.0:9080/sparql")
    dls_renderer = DLSyntaxObjectRenderer()
    kb_namespace = list(kb.ontology.classes_in_signature())[0].iri.get_namespace()
    expression_parser = DLSyntaxParser(kb_namespace)
    
    def tentris_predict(expr):
        start_time = time.time()
        retrieved = set([ind.str.split("/")[-1] for ind in tentris_kb.individuals(expression_parser.parse(expr))])
        time_taken = time.time() - start_time
        return retrieved, time_taken

    # Retrieval Results
    def concept_retrieval(expr):
        start_time = time.time()
        actual = set([ind.str.split("/")[-1] for ind in kb.individuals(expression_parser.parse(expr))])
        #print(f"Actual: {actual}")
        return actual, time.time() - start_time

    with open(args.dataset_dir+'/data/test_data.json') as f:
        concepts = json.load(f)
        concepts = list(map(expression_parser.parse, concepts))
    results = []
    counter = 0
    for expression in (tqdm_bar := tqdm(concepts, position=0, leave=True)):
        expr = dls_renderer.render(expression.get_nnf())
        retrieval_y, runtime_y = tentris_predict(expr)
        retrieval_kb_y, runtime_kb_y = concept_retrieval(expr)
        #print("Tentris")
        #print(retrieval_y)
        #print("KB")
        #print(retrieval_kb_y)
        #break
        jaccard_sim = jaccard_similarity(retrieval_y, retrieval_kb_y)
        # Compute the F1-score.
        f1_sim = f1_set_similarity(retrieval_y, retrieval_kb_y)
        # Store the data.
        results.append(
            {
                "Expression": owl_expression_to_dl(expression),
                "Type": type(expression).__name__,
                "Jaccard Similarity": jaccard_sim,
                "F1": f1_sim,
                "Runtime KB": runtime_kb_y,
                "Runtime Tentris": runtime_y,
                "Runtime Benefits": runtime_kb_y-runtime_y
            }
        )
        # Update the progress bar.
        tqdm_bar.set_description_str(
            f"Expression {counter}: {owl_expression_to_dl(expression)} | Jaccard Similarity:{jaccard_sim:.4f} | F1 :{f1_sim:.4f} | Runtime Benefits:{runtime_kb_y - runtime_y:.3f}"
        )
        counter += 1
    # Read the data into pandas dataframe
    df = pd.DataFrame(results)
    df = df[df["Type"] != "OWLClass"].reset_index(drop=True)
    # Save the experimental results into csv file.
    output_path_name = os.path.join(args.output_dir, "tentris_results.csv")
    df.to_csv(output_path_name, index=False)
    print("\n\x1b[6;30;42mSuccessfully saved the results!\x1b[0m\n")
    # Extract the numerical features.
    numerical_df = df.select_dtypes(include=["number"])
    mean_df = df[numerical_df.columns].mean()
    print(mean_df)

    def round_to_4(x):
        return f"{x:.4f}"

    with open(os.path.join(args.output_dir, f"tentris_avg_results.txt"), "w") as f:
        f.write(" & ".join(list(numerical_df.columns)) + "\n")
        f.write(" & ".join(list(map(round_to_4, mean_df.values.tolist()))) + "\n")

def get_default_arguments(description=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="The path of a folder containing training_data.json, which contains a list of class expressions or list of tuples where the first elements are class expressions."
                             ",e.g., datasets/carcinogenesis/")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="The location where to store the trained model and training results.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = get_default_arguments()
    execute(args)


if __name__ == "__main__":
    main()
