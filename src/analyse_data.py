import argparse
import os
import pickle
from collections import defaultdict
from os.path import exists

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
from transformers import set_seed

from probing_utils import extract_internal_reps_specific_layer_and_token, compile_probing_indices, \
    load_model_and_validate_gpu, get_probing_layer_names, LIST_OF_DATASETS, LIST_OF_MODELS, \
    MODEL_FRIENDLY_NAMES, LIST_OF_PROBING_LOCATIONS, compute_metrics_probing, prepare_for_probing,LIST_OF_MODELS_PATH


def parse_args_and_init_wandb():
    parser = argparse.ArgumentParser(
        description='Probe for hallucinations and create plots')
    parser.add_argument("--model", choices=LIST_OF_MODELS)
    parser.add_argument("--probe_at",
                        choices=LIST_OF_PROBING_LOCATIONS)
    # important args for the results of the papaer
    parser.add_argument("--seeds", nargs='+', type=int)
    parser.add_argument("--n_samples", help="size of validation data", default='all')
    parser.add_argument("--layer", type=int)
    parser.add_argument("--token", type=str)
    parser.add_argument("--save_clf", action='store_true', default=False, help="Whether to save the clf. If true, will look for a classifier before training and load it if exists.")
    parser.add_argument("--dataset", choices=LIST_OF_DATASETS, required=True)
    parser.add_argument("--test_dataset", choices=LIST_OF_DATASETS, required=False, default=None)

    args = parser.parse_args()
    os.environ["WANDB_MODE"] = "offline"
    if args.test_dataset is None:
        wandb.init(
            project="probe_hallucinations_specific",
            config=vars(args)
        )
    else:
        wandb.init(
            project="probe_hallucinations_generalization",
            config=vars(args)
        )

    return args

def main():
    args = parse_args_and_init_wandb()
    model_path=LIST_OF_MODELS_PATH[args.model]
    model, tokenizer = load_model_and_validate_gpu(model_path)
    data_test = None
    input_output_ids_test = None
    model_output_file = f"{args.dataset}/{MODEL_FRIENDLY_NAMES[args.model]}-answers-{args.dataset}.csv"
    data = pd.read_csv(model_output_file).reset_index()
    # input_output_ids = torch.load(
    #     f"{args.dataset}/{MODEL_FRIENDLY_NAMES[args.model]}-input_output_ids-{args.dataset}.pt")
    # # args.save_clf =True
    if args.test_dataset is not None:
        test_dataset = args.test_dataset
    else:
        test_dataset = args.dataset
    model_output_file_test = f"{args.dataset}/{MODEL_FRIENDLY_NAMES[args.model]}-answers-{args.dataset}_test.csv"
    load_test = False
    if os.path.isfile(model_output_file_test):
        data_test = pd.read_csv(model_output_file_test)
        # input_output_ids_test = torch.load(
        #     f"{args.dataset}/{MODEL_FRIENDLY_NAMES[args.model]}-input_output_ids-{test_dataset}_test.pt")
        load_test = True
    test_data_indices = data_test.index

    test_data_indices = test_data_indices[
            (data_test.iloc[test_data_indices]['valid_exact_answer'] == 1) & (
                    data_test.iloc[test_data_indices]['exact_answer'] != 'NO ANSWER') & (
                    data_test.iloc[test_data_indices]['exact_answer'].map(lambda x: type(x)) == str)]
    test=data_test.iloc[test_data_indices]
    print("all")
    model_output_file_test = f"{args.dataset}/answers_{args.dataset}_"
    seeds=[0,5,26,42,63]
    for seed in seeds:
        temp_test = model_output_file_test+str(seed)+".csv"
        if os.path.isfile(temp_test):
            data_test_1 = pd.read_csv(temp_test)
            test_data_indices = data_test_1.index
            if 'exact_answer' in data:
                test_data_indices = test_data_indices[
                    (data_test_1.iloc[test_data_indices]['automatic_correctness'] == 0)]
                test_d = data.iloc[test_data_indices]
                model_output_ = f"{args.dataset}/answers_{args.dataset}_wrong_proj_"
                temp = model_output_+str(seed)+".csv"
                pd.DataFrame.from_dict(test_d).to_csv(temp)
                print("1")

if __name__ == "__main__":
    main()