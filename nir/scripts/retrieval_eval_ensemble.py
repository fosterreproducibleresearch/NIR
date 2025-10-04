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
from ontolearn.knowledge_base import KnowledgeBase

from nir import InferenceDataset
from nir.config import NIRConfig
from nir.models import NIRComposite, NIRTransformer, NIRLSTM, NIRGRU
from nir.models.pmanet import PMAnet
from nir.utils import str2bool, read_embs_and_apply_agg
from nir.utils import score_all_inds, score_all_inds_composite

"""
How to run? E.g.,

python nir/scripts/retrieval_eval_ensemble.py --dataset_dir datasets/semantic_bible/ --output_dir Retrieval_semb_ensemble --models lstm gru transformer composite --pretrained_model_paths  new_train_dir/NIR_LSTM_semb/ new_train_dir/NIR_GRU_semb/ new_train_dir/NIR_Transformer_semb/ new_train_dir/NIR_Composite_semb/ --pma_model_path pma_pretrained/PMA_semb --th 0.5  
"""

def execute(args):
    dataset_dir = args.dataset_dir
    print(f"\n=================> Running {args.models} on {dataset_dir} <======================\n")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # Fix the random seed.
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    pma_net = None
    if any(model_name.lower()=="composite" for model_name in args.models):
        pma_net = PMAnet(NIRConfig().embedding_dim, NIRConfig().num_attention_heads, 1)
        pma_net.load_state_dict(
            torch.load(args.pma_model_path, map_location="cpu", weights_only=True))
        pma_net.eval()
        
    kb, all_individuals, embeddings = read_embs_and_apply_agg(args.dataset_dir, nn_agg=pma_net,
                                                              merge=True, complete_percent=args.complete_percent)
    if args.complete_percent is not None:
        complete_kb = KnowledgeBase(path=f"{args.dataset_dir}/kb/ontology.owl")
    else:
        complete_kb = None
    dls_renderer = DLSyntaxObjectRenderer()
    all_individuals_set = set(all_individuals)
    all_individuals_arr = np.array(sorted(all_individuals), dtype=object)
    all_ind_embs = torch.FloatTensor(
        embeddings.loc[all_individuals_arr].values)  # .to(device)
    kb_namespace = list(kb.ontology.classes_in_signature())[0].str
    if "#" in kb_namespace:
        kb_namespace = kb_namespace.split("#")[0] + "#"
    elif "/" in kb_namespace:
        kb_namespace = kb_namespace[:kb_namespace.rfind("/")] + "/"
    elif ":" in kb_namespace:
        kb_namespace = kb_namespace[:kb_namespace.rfind(":")] + ":"
    expression_parser = DLSyntaxParser(kb_namespace)

    AutoConfig.register("nir", NIRConfig)
    models = []

    for model_name, pretrained_model_path in zip(args.models, args.pretrained_model_paths):
        if model_name.lower() == "composite":
            #NIRConfig.batch_training = False
            AutoModel.register(NIRConfig, NIRComposite)
        elif model_name.lower() == "lstm":
            AutoModel.register(NIRConfig, NIRLSTM)
        elif model_name.lower() == "gru":
            AutoModel.register(NIRConfig, NIRGRU)
        elif model_name.lower() == "transformer":
            AutoModel.register(NIRConfig, NIRTransformer)
        model = AutoModel.from_pretrained(pretrained_model_path)
        tokenizer = None
        if model_name.lower() != "composite":
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
        print("\n\x1b[6;30;42mSuccessfully loaded a pretrained model!\x1b[0m\n")
        # Set a device for computations
        device = "cpu"
        if args.auto_detect_device:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
        model.eval()

        models.append((model, tokenizer))

    if args.auto_detect_device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        all_ind_embs = all_ind_embs.to(model.device)

    def nir_predict(model, expr, all_individuals_set, all_individuals_arr, all_ind_embs, embeddings, th):
        if isinstance(expr, str):
            expr = [expr]
        data = InferenceDataset(data=expr, all_individuals=all_individuals_set,
                                concept_to_instance_set=None,
                                embeddings=embeddings)
        
        expr, component_embeddings_dict = next(iter(data))
        component_embeddings_dict = [{key: val.to(device) for key, val in component_embeddings_dict.items()}]
        
        outputs = score_all_inds_composite(model, [expr], all_ind_embs, component_embeddings_dict).squeeze()
        return outputs

    def nir_encoders_predict(model, tokenizer, all_ind_embs, expr, all_individuals_arr, th):
        if isinstance(expr, str):
            expr = [expr]
        outputs = score_all_inds(model, tokenizer, all_ind_embs, expr, hidden_size=all_ind_embs.shape[1], chunk_size=args.chunk_size).squeeze()
        retrieved = set(all_individuals_arr[np.where(outputs > th)[0]])
        return outputs

    def ensemble_predict(models, expr, all_individuals_set, all_individuals_arr, all_ind_embs, embeddings, th):
        start_time = time.time()
        outputs = []
        for model, tokenizer in models:
            if tokenizer is None:
                scores = nir_predict(model, expr, all_individuals_set, all_individuals_arr, all_ind_embs, embeddings, th)                
            else:
                scores = nir_encoders_predict(model, tokenizer, all_ind_embs, expr, all_individuals_arr, th)
            outputs.append(scores)
        outputs = np.average(outputs, axis=0, weights=[0.15, 0.25, 0.55, 0.05])
        retrieved = set(all_individuals_arr[np.where(outputs > th)[0]])
        time_taken = time.time() - start_time
        return retrieved, time_taken
        
    # Retrieval Results
    def concept_retrieval(expr):
        start_time = time.time()
        actual = set([ind.str.split("/")[-1] for ind in kb.individuals(expression_parser.parse(expr))])
        #print(f"Actual: {actual}")
        return actual, time.time() - start_time

    def ground_truth_retrieval(expr):
        start_time = time.time()
        actual = set([ind.str.split("/")[-1] for ind in complete_kb.individuals(expression_parser.parse(expr))])
        #print(f"Actual: {actual}")
        return actual, time.time() - start_time

    with open(args.dataset_dir+'/data/test_data.json') as f:
        concepts = json.load(f)
        concepts = list(map(expression_parser.parse, concepts))
        
    results = []
    counter = 0
    for expression in (tqdm_bar := tqdm(concepts, position=0, leave=True)):
        expr = dls_renderer.render(expression.get_nnf())
        try:
            retrieval_y, runtime_y = ensemble_predict(models, expr, all_individuals_set, all_individuals_arr, all_ind_embs, embeddings, args.th)
        except Exception as e: # Catch any exception.
            print(str(e))
            raise
        if complete_kb is not None:
            retrieval_kb_y, runtime_kb_y = ground_truth_retrieval(expr)
            retrieval_incomplete_kb_y, runtime_incomplete_kb_y = concept_retrieval(expr)
            
            ## Compute metrics
            jaccard_sim = jaccard_similarity(retrieval_y, retrieval_kb_y)
            f1_sim = f1_set_similarity(retrieval_y, retrieval_kb_y)

            jaccard_sim_incomplete_kb = jaccard_similarity(retrieval_incomplete_kb_y, retrieval_kb_y)
            f1_sim_incomplete_kb = f1_set_similarity(retrieval_incomplete_kb_y, retrieval_kb_y)
        else:
            retrieval_kb_y, runtime_kb_y = concept_retrieval(expr)
            jaccard_sim = jaccard_similarity(retrieval_y, retrieval_kb_y)
            f1_sim = f1_set_similarity(retrieval_y, retrieval_kb_y)
            
        # Store the data.
        if complete_kb is not None:
            results.append(
                {
                    "Expression": owl_expression_to_dl(expression),
                    "Type": type(expression).__name__,
                    "Jaccard Similarity": jaccard_sim,
                    "F1": f1_sim,
                    "Jaccard Similarity Incomplete KB": jaccard_sim_incomplete_kb,
                    "F1 Incomplete KB": f1_sim_incomplete_kb
                }
            )
        else:
            results.append(
            {
                "Expression": owl_expression_to_dl(expression),
                "Type": type(expression).__name__,
                "Jaccard Similarity": jaccard_sim,
                "F1": f1_sim,
                "Runtime KB": runtime_kb_y,
                "Runtime NIR": runtime_y,
                "Runtime Benefits": runtime_kb_y-runtime_y,
                #"NIR_Retrieval": retrieval_y,
                #"KB Retrieval": retrieval_kb_y,
            }
        )
        # Update the progress bar.
        tqdm_bar.set_description_str(
            f"Expression {counter}: {owl_expression_to_dl(expression)} | Jaccard Similarity: {jaccard_sim:.4f} | F1 :{f1_sim:.4f} | Runtime Benefits: {runtime_kb_y - runtime_y:.3f}"
        )
        counter += 1
    # Read the data into pandas dataframe
    df = pd.DataFrame(results)
    df = df[df["Type"] != "OWLClass"].reset_index(drop=True)
    # Save the experimental results into csv file.
    output_path_name = os.path.join(args.output_dir, f"ensemble_results.csv")
    df.to_csv(output_path_name, index=False)
    print("\n\x1b[6;30;42mSuccessfully saved the results!\x1b[0m\n")
    # Extract the numerical features.
    numerical_df = df.select_dtypes(include=["number"])
    mean_df = df[numerical_df.columns].mean()
    print(mean_df)

    def round_to_4(x):
        return f"{x:.4f}"

    with open(os.path.join(args.output_dir, f"ensemble_avg_results.txt"), "w") as f:
        f.write(" & ".join(list(numerical_df.columns)) + "\n")
        f.write(" & ".join(list(map(round_to_4, mean_df.values.tolist()))) + "\n")

def get_default_arguments(description=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="The path of a folder containing training_data.json, which contains a list of class expressions or list of tuples where the first elements are class expressions."
                             ",e.g., datasets/carcinogenesis/")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="The location where to store the trained model and training results.")
    parser.add_argument("--caption", type=str, default="",
                        help="The caption to use for the table of results.")
    parser.add_argument("--th", type=float, default=0.5,
                        help="Threshold on probabilities to decide which individual is an instance of a class expression")
    parser.add_argument("--models", type=str, nargs="+",
                        default=["transformer", "lstm", "gru", "composite"],
                        help="List of neural instance retrievers")
    
    parser.add_argument("--auto_detect_device", type=str2bool, default=True,
                        help="Whether to automatically detect GPUs and use for computations")
    parser.add_argument("--use_pma", type=str2bool, default=True,
                        help="Whether to use PMA as encoder for atomic concepts via their sets of instances. This applies only to the `composite` model.")
    parser.add_argument("--pma_model_path", type=str, default=None,
                        help="Path to a pretrained PMA model.")
    parser.add_argument("--pretrained_model_paths", type=str, nargs="+", default=None,
                        help="Paths to a pretrained models, which must be of type `transformers.PretrainedModel`")
    parser.add_argument("--chunk_size", type=int, default=1024,
                        help="Batch size for scoring all individuals during instance retrieval")
    parser.add_argument("--complete_percent", type=int, default=None, help="Parameter specifying the size of the subsampled graph for experiments on incompleteness. Allowed values are `25, 50, or 75`", choices=[25, 50, 75])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = get_default_arguments()
    execute(args)


if __name__ == "__main__":
    main()
