import numpy as np    
from mpl_toolkits.axes_grid1 import host_subplot
import matplotlib.pyplot as plt
from Month_data import Month_all, Month_one, Month_duration,Month_all_duration, process_data
from cluster import k_means, choose_project
from MyPCA import pca_1, compute_newx, LinearReg, compute
import os
from sklearn.preprocessing import normalize
import csv
import math
from Entropy import entropy, get_weights


def trend(project_id, data, project_valid, scores, start, end):
    labels = ['forks','committer','commits','commit_comment',
        'req_opened','req_closed','req_merged','other','issue','issue_comment','watchers']
    ind = project_valid.index(project_id)
    scores_y = []
    for j in range(end-start):
        scores_y.append(scores[j][0])

    x = range(start, end)
    for cur in range(len(labels)):
        y = []
        for i in range(end-start):
            y.append(data[i][ind][cur]) 
        host = host_subplot(111)
        par = host.twinx()
        host.set_xlabel("month")
        host.set_ylabel("scores")
        par.set_ylabel(labels[cur])

        p1, = host.plot(x, scores_y, label="scores")
        p2, = par.plot(x, y, label=labels[cur])

        leg = plt.legend()
        host.yaxis.get_label().set_color(p1.get_color())
        leg.texts[0].set_color(p1.get_color())
        par.yaxis.get_label().set_color(p2.get_color())
        leg.texts[1].set_color(p2.get_color())
        plt.show()    

def get_data_score(project_id, data, project_valid, scaler, pca, start, end):
    scores = []
    ind = project_valid.index(project_id)
    for cur in range(end-start):
        print("转换前：", data[cur][ind])
        data_cur = scaler[cur].transform([data[cur][ind]])
        print("转换后：", data_cur)
        score_cur = compute(pca[cur], data_cur)
        scores.append(score_cur)
    return scores
    
def get_pca(data_proc, components_num):
    length = len(data_proc)
    pca = []
    for cur in range(length):
        pca_cur = pca_1(data_proc[cur], components_num)
        pca.append(pca_cur)  
    return pca

def trend_indicator(indicator, pca, start, end):
    x = range(start, end)
    weights = []
    labels = ['forks','committer','commits','commit_comment',
        'req_opened','req_closed','req_merged','other','issue','issue_comment','watchers']
    ind = labels.index(indicator)
    for cur in range(end-start):
        cur_ratio = pca[cur].explained_variance_ratio_
        cur_components_ = pca[cur].components_
        cur_weights = np.dot(cur_ratio, cur_components_)
        weights.append(cur_weights[ind])
    print("weights of "+ indicator + ': ', weights)
    plt.plot(x, weights)
    plt.xlabel("months")
    plt.ylabel("weights")
    plt.title(indicator)
    plt.show()

def compare(start, end, scores_1, scores_2):
    x = range(start, end)
    scores_y = []
    for j in range(end-start):
        scores_y.append(scores_1[j][0])    
    plt.plot(x, scores_y, label = 'My Score')
    plt.plot(x, scores_2, label = 'Her Score')
    plt.xlabel("months")
    plt.ylabel("scores")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    rootpath = os.getcwd() + '\\data\\'
    start, end = 1, 50
    project_id = 1486
    components_num = 10
    projects_valid, data = Month_all_duration(rootpath, start, end)
    data_proc, scalers = process_data(data, start, end)
    pca = get_pca(data_proc, components_num)
    scores = get_data_score(project_id, data_proc, projects_valid, scalers, pca, start, end)
    print("My scores: ", scores)
    weights = get_weights(data_proc, scalers, start, end, 11, 0.1)
    print("weights: ", weights)
    scores_2 = entropy(project_id, projects_valid, data_proc, weights, scalers, start, end)
    print("Her scores: ", scores_2)
    compare(start, end, scores, scores_2)
    # trend(project_id, data_proc, projects_valid, scores, start, end)
    # trend_indicator('forks', pca, start, end)
