import logging
from new_metric import calculate_ndcg
from simpletransformers.retrieval import RetrievalModel, RetrievalArgs
import pandas as pd


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


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
)

# Ideally we have to use an unseen data for evaluation (Possibly left out from eval.json to check the performance)
eval_data = pd.read_csv(f"data/data_dataset/dev_waa.tsv", sep="\t")
results, *_ = model.eval_model(
    eval_data,
    top_k_values=[1, 2, 3, 5, 10, 20, 100], ndcg=calculate_ndcg)
logging.info(results)
# Testing retrieval quality
docs, *_ = model.predict(["What is precision and recall?"])
logging.info(docs[0])