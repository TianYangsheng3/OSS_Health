import numpy as np
from Month_data import Month_all_duration, Month_duration, process_data
import math
import pandas as pd

def entropy(project_id, project_valid, data_proc, weights, scalers, start, end):
    ind = project_valid.index(project_id)
    scores = []
    length = len(project_valid)

    for cur in range(end-start):
        # print("her pre: ", data_proc[cur][ind])
        # data_cur = scalers[cur].transform([data_proc[cur][ind]])
        # print("her post: ", data_cur)
        score_cur = np.dot(weights[cur], np.array(data_proc[cur][ind]))
        scores.append(score_cur)
    return scores


def get_weights(data_proc, scalers, start, end, feature_nums, e):
    weights = []
    labels = ['forks','committer','commits','commit_comment',
        'req_opened','req_closed','req_merged','other','issue','issue_comment','watchers']    
    for cur in range(end-start):
        cur_weights = []
        data = pd.DataFrame(scalers[cur].transform(data_proc[cur]), columns=labels)
        C = 0
        for i in range(len(labels)):
            sumx = data[labels[i]].sum()
            Ci = 0
            sumCi = 0
            for j in range(len(data)):
                if data[labels[i]][j]==0:
                    Pij = e/(len(data)*e+sumx)
                elif data[labels[i]][j]>0:
                    Pij = data[labels[i]][j]/sumx
                else:
                    print("Error in compute Pij")
                    return
                Ci += (Pij * math.log(Pij))
            Ci = (1/math.log(len(data)))*Ci
            C += Ci
            cur_weights.append(Ci)
        for j in range(len(cur_weights)):
            cur_weights[j] = cur_weights[j]/C
        weights.append(cur_weights)
    return np.array(weights)
            
  

if __name__ == '__main__':
    pass