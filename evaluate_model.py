from asyncio.log import logger
from turtle import title
import pandas as pd
import csv
import logging
import numpy as np
import os
import argparse
from new_metric import calculate_ndcg, calculate_recall_100
from simpletransformers.retrieval import RetrievalModel, RetrievalArgs

parser = argparse.ArgumentParser()
parser.add_argument("--ablation", help="To perform ablation experiment on the query encoder. Changes the output-dir based on the layers", type=bool, default=False)
parser.add_argument("--query_layers", help="To perform ablation. Remove layers in query encoder. Default is 12", type=int, default=12)
args = parser.parse_args()

dataset_dir = "dataset"
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

logging.info("Loading doc data...")
# Read the data from the CSV :TODO Run the modify_dataset.py before running this file

# Path to the trained model
model_type = "custom"
# Generating index only for the DPR for now. Can switch later for other research questions
model_name = "output/bert-base-uncased"

# Path to the Wikipedia passage collection
index_path = "data/data_dataset/corpus_to_index.tsv"

model_args = RetrievalArgs()
model_args.output_dir = f"output/{model_name}"
model_args.include_title = False
# Loading the model automatically builds the index
if args.ablation:
    logging.info(f"Performing Ablation experiment with Query layer = {args.query_layers}")
    # model_args.best_model_dir = f"output/{question_name}_{args.query_layers}_new/best_model"
    # model_args.output_dir = f"output/{question_name}_{args.query_layers}_new"

model = RetrievalModel(
    model_type=model_type,
    model_name=model_name,
    args=model_args,
    prediction_passages=index_path,
    query_static_embeddings=args.ablation,
    query_num_layers=args.query_layers
)

# Ideally we have to use an unseen data for evaluation (Possibly left out from eval.json to check the performance)
eval_data = pd.read_csv(f"data/data_dataset/dev_waa.tsv", sep="\t")


results, *_ = model.eval_model(eval_data, top_k_values=[1, 2, 3, 5, 10, 20, 100], output_dir = f"output/bert-base-uncased_static_{args.query_layers}" if args.ablation else f"output/bert-base-uncased_static", ndcg=calculate_ndcg, recall_100=calculate_recall_100)
logging.info(results)

import time
start_time=time.time()
docs, *_ = model.predict(["What is precision and recall?"])
logging.info(docs)
logging.info(f"It took {time.time()-start_time} ms to predict 1 query")