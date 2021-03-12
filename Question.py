import Trend
import Entropy
import random
import os, csv
from Month_data import Month_duration, Month_all_duration, process_data
from scipy import stats
import numpy as np
import MyPCA
import matplotlib.pyplot as plt

def Question_1(data_proc, projects_valid, scalers, weights, pca, start, end, have_std):
    # 随机选择20（10）个项目，画出每个项目两种方法的对比图
    filepath = 'Question_1'
    selects_pid = random.sample(projects_valid, 20)
    cnt = 0
    indicator_num = 11
    if have_std:
        indicator_num = 22
    for pid in projects_valid: #selects_pid
        scores_pca = Trend.get_data_score(pid, data_proc, projects_valid, scalers, pca, start, end)
        scores_entropy = Entropy.entropy(pid, projects_valid, data_proc, weights, scalers, start, end)
        Trend.compare(pid, start, end, scores_pca, scores_entropy, filepath)

    # 显示项目的健康性与各指标的变化趋势图（还是做相关系数表）
        Trend.trend(pid, data_proc, projects_valid, scores_pca, start, end, filepath, have_std=have_std)
        Trend.correlation(pid, data_proc, projects_valid, scores_pca, start, end, cnt, have_std)
        cnt += 1

    # 显示满足t-test零假设结果的项目个数   
    Trend.t_test_data(projects_valid, data_proc, weights, scalers, pca, start, end)

def Question_2(data_proc, projects_valid, scalers, pca, start, end):
    ### 观察所有项目从start到end月的健康性频率分布直方图
    scores_all_proj = Trend.get_scores_projects(data_proc, projects_valid, scalers, pca, start, end)
    Trend.health_hist(scores_all_proj, start, end)

    ### 得到每个健康性分布区间的项目个数
    Trend.interval_stats(scores_all_proj, start, end)


def Question_3(data_proc, projects_valid, scalers, pca, start, end, have_std,interval):

    selects = [[] for i in range(len(interval))]
    scores_all_proj = Trend.get_scores_projects(data_proc, projects_valid, scalers, pca, start, end)
    
    for i in range(len(projects_valid)):
        cur_scores = []
        for cur_month in range(end-start):
            cur_scores.append(scores_all_proj[cur_month][i])
        cur_scores = np.array(cur_scores)
        avg = cur_scores.sum()/(end-start)
        for j in range(len(interval)):
            if avg<=interval[j]:
                selects[j].append(projects_valid[i])
                break
    results = Trend.Avg_value_interval(data_proc, scores_all_proj,  projects_valid, selects, start, end, have_std, interval)
    Trend.compare_interval(results, start, end, have_std, interval)


### observer the trend of indicators and health from start to end.
def Question_4(pca, start, end, have_std):
    labels = ['forks','contributor','commits','commit_comment',
        'req_opened','req_closed','req_merged','other','issue','issue_comment','watchers']
    for indicator in labels:
        Trend.trend_indicator(indicator, pca, start, end, have_std)

if __name__=='__main__':
    labels = ['forks','committer','commits','commit_comment',
        'req_opened','req_closed','req_merged','other','issue','issue_comment','watchers']
    rootpath = os.getcwd() + '\\data\\'
    start, end = 1, 50
    components_num = 10
    have_std = True  

    ### 得到数据
    projects_valid, data = Month_all_duration(rootpath, start, end, have_std = have_std)
    data_proc, scalers = process_data(data, start, end)
    ### 得到每个月的pca权重
    pca = Trend.get_pca(data_proc, scalers, components_num)
    # scores = Trend.get_data_score(1486, data_proc, projects_valid, scalers, pca, start, end)
    ### 得到每个月的信息熵方法的权重
    weights = Entropy.get_weights(data_proc, scalers, start, end, have_std = have_std)    


    Question_1(data_proc, projects_valid, scalers, pca, weights, start, end, have_std = have_std)
    Question_2(data_proc, projects_valid, scalers, pca, start, end)

    interval = [0.05, 0.15, 1]
    Question_3(data_proc, projects_valid, scalers, pca, start, end, have_std, interval)
    Question_4(pca, start, end, have_std)
