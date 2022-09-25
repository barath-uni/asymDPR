import json
from os import sep
import pandas as pd
from tqdm.auto import tqdm

dataset_dir = "dataset"

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
        query_dict["gold_passage"].append([passage["passage_text"] for passage in passages[key]][0])
        query_dict["query_text"].append(questions[key])
        answer = answers.get(key, [])
        query_dict["query_id"].append(query_ids[key])
    df = pd.DataFrame(query_dict).dropna()
    
    # Random sample to see if this runs the script faster
    df = df.sample(n=16000) 
    print(len(df.index))
    return df

with open(f"{dataset_dir}/train_v2.1.json", 'r') as f:
    train_data = json.load(f)

with open(f"{dataset_dir}/dev_v2.1.json", 'r') as f:
    dev_data = json.load(f)

#train_df = change_data(train_data)
#dev_df = change_data(dev_data)
# We will replace the tsvs depending on the n. This shouldn't be a problem
#train_df.to_csv(f"data/data_{dataset_dir}/train.tsv", sep="\t", index=False)
#dev_df.to_csv(f"data/data_{dataset_dir}/dev.tsv", sep="\t", index=False)

print("DONE CHANGING THE DATA for TRAIN & DEV")

# Seeing how the train_data and dev_data size changes for only wellformedanswers

def makewf(input,output):
    df = pd.read_json(input)
    df = df.drop('answers',1)
    df = df.drop('passages',1)
    df = df.drop('query_type',1)
    df = df.rename(columns={'wellFormedAnswers':'gold_passage'})
    df = df[df.gold_passage != '[]']
    df['gold_passage'] = df['gold_passage'].map(lambda x: x[0])
    df = df.rename(columns={'query': 'query_text'})
    # sample only 50% of the gold passage considering the computational constraints
    df = df.sample(frac=0.5)
    print("well formed answers stats")
    print(len(df.index))
    print(df['gold_passage'])
    return df

def generate_passage(dataframes):
    df = df.concat(dataframes)
    df = df.drop('query_text',1)
    df = df.drop('query_text',1)
    df = df.rename(columns={'gold_passage':'passages', 'query_id':'corpus_id'})

train_df_wfa = makewf(f"{dataset_dir}/train_v2.1.json", "")
train_df_wfa.to_csv(f"data/data_{dataset_dir}/train_waa.tsv", sep="\t", index=False)
dev_df_wfa = makewf(f"{dataset_dir}/dev_v2.1.json", "")
dev_df_wfa.to_csv(f"data/data_{dataset_dir}/dev_waa.tsv", sep="\t", index=False)

# Changes to generate corpus for indexing
train_dev_passages = generate_passage([dev_df_wfa, train_df_wfa])
train_dev_passages.to_csv(f"data/data_{dataset_dir}/corpus_to_index.tsv", index=False)