import pandas as pd
import logging
import os
from simpletransformers.retrieval import RetrievalModel, RetrievalArgs
import argparse
from new_metric import calculate_ndcg, calculate_recall_100
from transformers import BertPreTrainedModel, AutoConfig, AutoModel, AutoTokenizer, PretrainedConfig
import torch

class ExtendedTransformer(BertPreTrainedModel):
    
    def __init__(self, config, model_name):
        self.model_type = model_name
        super().__init__(config)
        self.bert = AutoModel.from_pretrained(model_name)
        self.linear = torch.nn.Linear(self.config.hidden_size, 768, bias=False)
        self.post_init()


# Some parameters to train withput changing the script everytime
parser = argparse.ArgumentParser()
parser.add_argument("--query_model", help="A valid hugging face BERT based model.", type=str, default="bert-base-uncased")
parser.add_argument("--ablation", help="To perform ablation experiment on the query encoder. Changes the output-dir based on the layers", type=bool, default=False)
parser.add_argument("--query_layers", help="To perform ablation. Remove layers in query encoder. Default is 12", type=int, default=12)
parser.add_argument("--passage_model", help="A valid hugging face BERT based model.", type=str, default="bert-base-uncased")
parser.add_argument("--batch_size", help="Control the batch size for the DPR to train.", type=int, default=40)
args = parser.parse_args()



dataset_dir = "dataset"
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

logging.info("Loading doc data...")
# Read the data from the CSV :TODO Run the modify_dataset.py before running this file

train_data = pd.read_csv(f"data/data_{dataset_dir}/train_waa.tsv", sep="\t")
eval_data = pd.read_csv(f"data/data_{dataset_dir}/dev_waa.tsv", sep="\t")

logging.info(f"Starting to train with {args.query_model}")

model_type = "custom"
model_name = None
context_name = args.passage_model
# Using a different encoder to train and get the performance
question_name = args.query_model

model_args = RetrievalArgs()

# Training parameters
model_args.num_train_epochs = 40
model_args.train_batch_size = args.batch_size
model_args.learning_rate = 1e-5
model_args.max_seq_length = 256

# Evaluation parameters
model_args.retrieve_n_docs = 100
model_args.eval_batch_size = 100
model_args.evaluate_during_training = True
model_args.evaluate_during_training_verbose = True
# model_args.evaluate_during_training_steps = 200
model_args.include_title = False
# Model tracking
# model_args.wandb_project = "Dense retrieval with Simple Transformers"
model_args.save_model_every_epoch = False
model_args.save_eval_checkpoints = False
model_args.save_steps = -1
model_args.save_best_model = True

# Local change to test the metric, will be removed
model_args.best_model_dir = f"output/{question_name}_new/best_model"
model_args.output_dir = f"output/{question_name}_new"
if args.ablation:
    logging.info(f"Performing Ablation experiment with Query layer = {args.query_layers}")
    model_args.best_model_dir = f"output/{question_name}_{args.query_layers}_new/best_model"
    model_args.output_dir = f"output/{question_name}_{args.query_layers}_new"
# We dont want to accidentally remove an already run model, so keeping it as False which should help adding a new output dir name
model_args.overwrite_output_dir = True
# if args.query_model != "bert-base-uncased" or args.query_model != "distilbert-base-uncased":
#     if not os.path.exists("output/{question_name}_ExtendedModel"):
#         config = AutoConfig.from_pretrained(question_name)
#         # Have to manually store the tokenizer so simpletransformers can pick this up
#         tokenizer = AutoTokenizer.from_pretrained(question_name)
#         new_model = ExtendedTransformer(config, question_name)
#         logging.info(new_model.config.to_dict())
#         new_model.save_pretrained(save_directory=f"output/{question_name}_ExtendedModel")
#         tokenizer.save_pretrained(save_directory=f"output/{question_name}_ExtendedModel")
#         # Additionally have to change the config.name_or_path variable
#         # config.name_or_path = f"output/{question_name}_ExtendedModel"
#         # config.save_pretrained(save_directory=f"output/{question_name}_ExtendedModel")
#         logging.info(question_name)
#     # Adds an MLP to convert the projection dimension to match the bert-base-uncased dimension
    # question_name = f"output/{question_name}_ExtendedModel"
logging.info(f"Loading the model from {question_name}")
model = RetrievalModel(
    model_type=model_type,
    model_name=model_name,
    context_encoder_name=context_name,
    query_encoder_name=question_name,
    args=model_args,
)

# Including the metric that is needed, for now adding ndcg
model.train_model(train_data, eval_data=eval_data, output_dir = f"output/{question_name}_{args.query_layers}" if args.ablation else f"output/{question_name}_new" , ndcg=calculate_ndcg, recall_100=calculate_recall_100)
