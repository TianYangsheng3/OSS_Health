import numpy as np
import math
import pandas as pd

def entropy(project_id, project_valid, data, weights, scalers, start, end):
    ind = project_valid.index(project_id)
    data_proc = data.copy()
    scores = []
    length = len(project_valid)

    for cur in range(end-start):
        # print("转换前her pre: ", data_proc[cur][ind])
        data_cur = scalers[cur].transform([data_proc[cur][ind]])
        # print("转换后her post: ", data_cur)
        score_cur = np.dot(weights[cur], np.array(data_cur[0]))
        scores.append(score_cur)
    return scores


def get_weights(data_, scalers, start, end, e = 0.1, have_std = False):
    weights = []
    data_proc = data_.copy()
    labels_mean = ['forks','committer','commits','commit_comment',
        'req_opened','req_closed','req_merged','other','issue','issue_comment','watchers']
    labels_std = ['forks_std','committer_std','commits_std','commit_comment_std',
        'req_opened_std','req_closed_std','req_merged_std','other_std','issue_std','issue_comment_std','watchers_std']
    if have_std:
        labels = labels_mean + labels_std
    else:
        labels = labels_mean
    # print("data shape: ", data_proc.shape)
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