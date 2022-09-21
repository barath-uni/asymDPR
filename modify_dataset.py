import json
import pandas as pd
from tqdm.auto import tqdm

dataset_dir = ""

def change_data(data):
    query_dict = {
        "query_text": [],
        "gold_passage": [],
        "query_id": [],
    }

    questions = data["query"]
    answers = data.get("answers", {})
    passages = data["passages"]
    query_ids = data["query_id"]
    for key in questions:

        query_dict["gold_passage"] = [passage["passage_text"] for passage in passages[key]]
        query_dict["query_text"] = questions[key]
        answer = answers.get(key, [])
        query_dict["query_id"] = query_ids[key]
    df = pd.DataFrame(query_dict).dropna()
    return df

with open(f"{dataset_dir}/train_v2.1.json", 'r') as f:
    train_data = json.load(f)

with open(f"{dataset_dir}/dev_v2.1.json", 'r') as f:
    dev_data = json.load(f)

train_df = change_data(train_data)
dev_df = change_data(dev_data)
train_df.to_csv(f"data/data_{dataset_dir}/train.tsv", sep="\t", index=False)
dev_df.to_csv(f"data/data_{dataset_dir}/dev.tsv", sep="\t", index=False)

print("DONE CHANGING THE DATA for TRAIN & DEV")