from asyncio.log import logger
from turtle import title
import pandas as pd
import csv
import logging
import numpy as np
import os
from simpletransformers.retrieval import RetrievalModel, RetrievalArgs

dataset_dir = "dataset"
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

logging.info("Loading doc data...")
# Read the data from the CSV :TODO Run the modify_dataset.py before running this file

train_data = pd.read_csv(f"data/data_{dataset_dir}/train_waa.tsv", sep="\t")
eval_data = pd.read_csv(f"data/data_{dataset_dir}/dev_waa.tsv", sep="\t")
model_type = "custom"
model_name = None
context_name = "bert-base-uncased"
question_name = "bert-base-uncased"
model_args = RetrievalArgs()

# Training parameters
model_args.num_train_epochs = 40
model_args.train_batch_size = 40
model_args.learning_rate = 1e-5
model_args.max_seq_length = 256

# Evaluation parameters
model_args.retrieve_n_docs = 100
model_args.eval_batch_size = 100
model_args.evaluate_during_training = False
model_args.evaluate_during_training_verbose = True
# model_args.evaluate_during_training_steps = 200
model_args.include_title = False
# Model tracking
# model_args.wandb_project = "Dense retrieval with Simple Transformers"
model_args.save_model_every_epoch = False
model_args.save_eval_checkpoints = False
model_args.save_steps = -1
model_args.save_best_model = True
model_args.overwrite_output_dir = True

model = RetrievalModel(
    model_type=model_type,
    model_name=model_name,
    context_encoder_name=context_name,
    query_encoder_name=question_name,
    args=model_args,
)


# Ideally this should also index, because according to doc the additional_eval_passages performs an indexing before testing.
# But not sure since i cannot run the whole training locally
model.train_model(train_data, output_dir = "output/dpr")
