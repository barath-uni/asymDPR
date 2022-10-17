from asyncio.log import logger
from turtle import title
import pandas as pd
import csv
import logging
import numpy as np
import os
from new_metric import calculate_ndcg, calculate_recall_100
from simpletransformers.retrieval import RetrievalModel, RetrievalArgs

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
model = RetrievalModel(
    model_type=model_type,
    model_name=model_name,
    args=model_args,
    prediction_passages=index_path,
    # query_static_embeddings=True
)

# Ideally we have to use an unseen data for evaluation (Possibly left out from eval.json to check the performance)
eval_data = pd.read_csv(f"data/data_dataset/dev_waa.tsv", sep="\t")


results, *_ = model.eval_model(eval_data, top_k_values=[1, 2, 3, 5, 10, 20, 100], output_dir = f"output/bert-base-uncased_static", ndcg=calculate_ndcg, recall_100=calculate_recall_100)
logging.info(results)