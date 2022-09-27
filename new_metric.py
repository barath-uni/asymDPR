import logging
import numpy as np
import string
# Need  a metric calculation function for NDCG
def calculate_ndcg(gold_passages, doc_texts):
    logging.info(len(doc_texts))
    logging.info(len(doc_texts[0]))
    relevance_list = np.zeros((len(gold_passages), len(doc_texts[0])))
    for i, (docs, truth) in enumerate(zip(doc_texts, gold_passages)):
        for j, d in enumerate(docs):    
            if d.strip().lower().translate(
                str.maketrans("", "", string.punctuation)
            ) == truth.strip().lower().translate(
                str.maketrans("", "", string.punctuation)
            ):
                relevance_list[i, j] = 1
                break
    logging.info(relevance_list)
    return 0.0
    # logging.info("NDCG METRIC", total_rel/den_count)
    # return total_rel/den_count

def calculate_recall_100(gold_passages, doc_texts):
    act_set = set(gold_passages)
    # Hardcoding K for Recall@100
    pred_set = set(doc_texts[:100])
    result = len(act_set.intersection(pred_set))/float(len(act_set))
    return result