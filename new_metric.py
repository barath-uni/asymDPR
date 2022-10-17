import logging
import numpy as np
import string
from sklearn.metrics import ndcg_score


# Need  a metric calculation function for NDCG
def calculate_ndcg(gold_passages, doc_texts):
    logging.info(len(doc_texts))
    logging.info(len(doc_texts[0]))
    y_truth = np.zeros((len(gold_passages), len(doc_texts[0])))
    relevance_list = np.zeros((len(gold_passages), len(doc_texts[0])))
    for i in range(len(gold_passages)):
        y_truth[i][0] = 1
    for i, (docs, truth) in enumerate(zip(doc_texts, gold_passages)):
        for j, d in enumerate(docs):    
            if d.strip().lower().translate(
                str.maketrans("", "", string.punctuation)
            ) == truth.strip().lower().translate(
                str.maketrans("", "", string.punctuation)
            ):
                relevance_list[i, j] = 1
                break
    logging.info("RELEVANCE SCORE")
    logging.info(relevance_list)
    logging.info("Y TRUTH")
    logging.info(y_truth)
    return ndcg_score(y_truth, relevance_list)

def calculate_recall_100(gold_passages, doc_texts):
    act_set = set(gold_passages)
    # Hardcoding K for Recall@100
    logging.info("DOC TEXTS AT 100")
    # logging.info([doc_texts[:,:100]])
    # pred_set = set(doc_texts[:,:100])
    # # logging.info(pred_set)
    # result = len(act_set.intersection(pred_set))/float(len(act_set))
    # return result
    return 0.0