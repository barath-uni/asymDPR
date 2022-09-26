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
        if gold_passages[id].strip().lower().translate(str.maketrans("", "", string.punctuation)) == row:
            rel += 1 / np.log2(id_count + 1)
        last_id = id

    total_rel += rel
    den_count += 1

    logging.info("NDCG METRIC", total_rel/den_count)
    return total_rel/den_count

# Need a metric calculation function for Recall@k

# 