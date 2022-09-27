import logging
import numpy as np
import string
# Need  a metric calculation function for NDCG
def calculate_ndcg(gold_passages, doc_texts):
    last_id = -1
    den_count = 0
    total_rel = 0
    for id, row in enumerate(doc_texts):
        if last_id != id:
            if last_id != -1:
                total_rel += rel
                den_count += 1
            id_count = 1
            rel = 0
        else:
            id_count += 1
        logging.info(gold_passages[id])
        logging.info(row)
        if gold_passages[id].strip().lower().translate(str.maketrans("", "", string.punctuation)) == row.strip().lower().translate(str.maketrans("", "", string.punctuation)):
            rel += 1 / np.log2(id_count + 1)
        last_id = id

    total_rel += rel
    den_count += 1

    logging.info("NDCG METRIC", total_rel/den_count)
    return total_rel/den_count

def calculate_recall_100(gold_passages, doc_texts):
    act_set = set(gold_passages)
    # Hardcoding K for Recall@100
    pred_set = set(doc_texts[:100])
    result = len(act_set.intersection(pred_set))/float(len(act_set))
    return result