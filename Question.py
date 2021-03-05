import Trend
import Entropy
import random
import os, csv
from Month_data import Month_duration, Month_all_duration, process_data
from scipy import stats


def Question_1(data_proc, projects_valid, scalers, pca, weights, start, end, have_std):
    # 随机选择20（10）个项目，画出每个项目两种方法的对比图
    filepath = 'Question_1'
    selects_pid = random.sample(projects_valid, 20)
    cnt = 0
    for pid in selects_pid:
        scores_pca = Trend.get_data_score(pid, data_proc, projects_valid, scalers, pca, start, end)
        scores_entropy = Entropy.entropy(pid, projects_valid, data_proc, weights, scalers, start, end)
        Trend.compare(pid, start, end, scores_pca, scores_entropy, filepath)

    # 显示项目的健康性与各指标的变化趋势图（还是做相关系数表）
        Trend.trend(pid, data_proc, projects_valid, scores_pca, start, end, filepath, have_std=have_std)
        Trend.correlation(pid, data_proc, projects_valid, scores_pca, start, end, cnt, have_std)
        cnt += 1
    # 显示满足t-test零假设结果的项目个数   
    Trend.t_test_data(projects_valid, data_proc, weights, scalers, pca, start, end)




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
