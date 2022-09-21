import json

with open("train_v2.1.json", "r") as f:
    train = json.load(f)

print(train.keys())

