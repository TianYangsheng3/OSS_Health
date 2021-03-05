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
from scipy import stats


### 观察项目project_id的健康性与各个评价指标的变化趋势
def trend(project_id, data, project_valid, scores, start, end, filepath, have_std):
    labels = ['forks','contributor','commits','commit_comment',
        'req_opened','req_closed','req_merged','other','issue','issue_comment','watchers']
    ind = project_valid.index(project_id)
    scores_y = []
    for j in range(end-start):
        scores_y.append(scores[j][0])

    x = range(start, end)
    for cur in range(len(labels)):
        y = []
        y_2 = []
        for i in range(end-start):
            y.append(data[i][ind][cur]) 
            if have_std:
                y_2.append(data[i][ind][cur+11])    # 标准差数据
        host = host_subplot(111)
        par = host.twinx()
        host.set_xlabel("month")
        host.set_ylabel("values")
        if have_std:
            par.set_ylabel(labels[cur])      # 标准差数据

        p1, = host.plot(x, scores_y, label="Health")
        p2, = par.plot(x, y, label=labels[cur]+'_mean')
        if have_std:
            p3, = par.plot(x, y_2, label = labels[cur]+'_std')   # 标准差数据

        leg = plt.legend()
        host.yaxis.get_label().set_color(p1.get_color())
        leg.texts[0].set_color(p1.get_color())
        par.yaxis.get_label().set_color(p2.get_color())
        leg.texts[1].set_color(p2.get_color())
        if have_std:
            leg.texts[2].set_color(p3.get_color())  # 标准差数据
        plt.title("The Change Trend of Health and Evaluation Indicators of Project id: "+str(project_id))
        f_path = "fig\\" + filepath + "\\trend_"+labels[cur]+"_project_"+str(project_id)+".pdf"
        plt.savefig(f_path, bbox_inches = 'tight')
        # plt.show()   
        plt.close() 

### 得到项目project_id从start到end月的健康性
def get_data_score(project_id, data, project_valid, scaler, pca, start, end):
    scores = []
    data_proc = data.copy()
    ind = project_valid.index(project_id)
    for cur in range(end-start):
        # print("转换前：", data_proc[cur][ind])
        data_cur = scaler[cur].transform([data_proc[cur][ind]])
        # print("转换后：", data_cur)
        score_cur = compute(pca[cur], data_cur)
        scores.append(score_cur)
    return scores


def get_pca(data, scalers, components_num):
    data_proc = data.copy()
    length = len(data_proc)
    pca = []
    for cur in range(length):
        data_proc[cur] = scalers[cur].transform(data_proc[cur])
        pca_cur = pca_1(data_proc[cur], components_num)
        pca.append(pca_cur)  
    return pca


### 观察使用pca方法时，评价指标indicator从start到end月的变化趋势
def trend_indicator(indicator, pca, start, end, have_std):
    x = range(start, end)
    weights = []
    weights_std = []        # 标准差数据
    labels = ['forks','contributor','commits','commit_comment',
        'req_opened','req_closed','req_merged','other','issue','issue_comment','watchers']
    ind = labels.index(indicator)
    for cur in range(end-start):
        cur_ratio = pca[cur].explained_variance_ratio_
        cur_components_ = pca[cur].components_
        cur_weights = np.dot(cur_ratio, cur_components_)
        sum_v = cur_weights.sum()
        for i in range(len(cur_weights)):
            cur_weights[i] = cur_weights[i]/sum_v
        weights.append(cur_weights[ind])
        if have_std:
            weights_std.append(cur_weights[ind+11]) # 标准差数据
    # print("weights of "+ indicator + ': ', weights)
    plt.plot(x, weights, label = "mean")
    if have_std:
        plt.plot(x, weights_std, label = "std")    # 标准差数据
    plt.xlabel("months")
    plt.ylabel("weights")
    plt.title(indicator)
    plt.legend()
    plt.show()


### 比较项目project_id的从start到end月的两种方法健康性
def compare(project_id, start, end, scores_1, scores_2, filepath):
    x = range(start, end)
    scores_y = []
    for j in range(end-start):
        scores_y.append(scores_1[j][0])    
    plt.plot(x, scores_y, label = 'Health of PCA')
    plt.plot(x, scores_2, label = 'Health of Entropy')
    plt.xlabel("months")
    plt.ylabel("values")
    plt.title('Comparison of PCA and Entropy: Project id '+str(project_id))
    plt.legend()
    f_path = "fig\\" + filepath + "\\compare_project_"+str(project_id)+".pdf"
    plt.savefig(f_path, bbox_inches = 'tight')
    # plt.show()
    plt.close()


### 得到所有项目采用pca方法从start到end月的健康性
def get_scores_projects(data, project_valid, scaler, pca, start, end):
    scores = [[] for i in range(end-start)]
    for project_id in project_valid:
        cur_score = get_data_score(project_id, data, project_valid, scaler, pca, start, end)
        for i in range(end-start):
            scores[i].append(cur_score[i][0])
    # print("scores_all_projects: ", scores)
    return scores

### 健康性直方图
def health_hist(scores, start, end):
    scores = np.array(scores)
    print("scores shape: ", scores.shape)
    for i in range(end-start):
        fig, ax = plt.subplots()
        num_bins = 20
        # weights = np.ones_like(scores[i])/float(len(scores[i]))
        # counts, num_bins = np.histogram(scores[i])
        # n, bins, patches = ax.hist(scores[i], num_bins, density = True, weights=weights)   # 频率直方图
        n, bins, patches = ax.hist(scores[i], num_bins)         # 频数直方图

        ax.set_xlabel('Scores')
        ax.set_ylabel('Numbers')
        ax.set_title('health scores of projects of '+ str(start+i) +  '-th month')
        print(str(start+i) + "-th month n: ", n)

        def autolabel(rects):
            for rect in rects:
                value = rect.get_height()
                if value>0:
                    ax.text(rect.get_x()+rect.get_y()/2., value+2, '%d' % value)
                    # ax.text(rect.get_x()+rect.get_y()/2., value*1.02, '%s' % float(round(value*(np.diff(bins)[0]),3)))  
        
        autolabel(patches)
        fig.tight_layout()
        plt.show()

###  result of t_test
def t_test_data(projects_valid, data_proc, weights, scalers, pca, start, end):
    to_file = 'file\\t_test.csv'
    header = ['project_id', 'method']
    count_sum = len(projects_valid)
    count_n = 0
    for i in range(start, end):
        header.append('month_'+str(i+1))
    with open(to_file, 'w', newline="") as f:
        f_csv = csv.writer(f)
        f_csv.writerow(header)
        for p_id in projects_valid:
            scores_0_tmp = get_data_score(p_id, data_proc, projects_valid, scalers, pca, start, end)
            scores_1 = entropy(p_id, projects_valid, data_proc, weights, scalers, start, end)
            scores_0 = []
            for i in range(end-start):
                scores_0.append(scores_0_tmp[i][0])
            f_csv.writerow([p_id, 'PCA']+scores_0)
            f_csv.writerow([p_id, 'Entropy']+scores_1)
            _ , p = stats.ttest_ind(np.array(scores_0), np.array(scores_1))
            if p<0.05:
                count_n += 1
        print("t-test result: "+str(count_sum)+", "+str(count_n))
                
### 将相关系数写入到文件中
def correlation(project_id, data_proc, project_valid, scores, start, end, cnt, have_std):
    filepath = 'fig\\Question_1\\correlation.csv'
    header_mean = ['project_id', 'correlation', 'forks_mean','contributor_mean','commits_mean','commit_comment_mean',
        'req_opened_mean','req_closed_mean','req_merged_mean','req_other_mean','issue_mean','issue_comment_mean','watchers_mean']
    header_std = ['forks_std','contributor_std','commits_std','commit_comment_std',
        'req_opened_std','req_closed_std','req_merged_std','req_other_std','issue_std','issue_comment_std','watchers_std']
    if have_std:
        header = header_mean+header_std
        num = len(header_std)*2
    else:
        header = header_mean
        num = len(header_std)

    ind = project_valid.index(project_id)
    scores_y = []
    for j in range(end-start):
        scores_y.append(scores[j][0])

    with open(filepath, 'a+') as f:
        f_csv = csv.writer(f)
        if cnt==0:
            f_csv.writerow(header)

        v = []
        p = []
        for cur in range(num):
            cur_data = []
            for i in range(end-start):
                cur_data.append(data_proc[i][ind][cur]) 
            v_cur, p_cur = stats.pearsonr(scores_y, cur_data)
            v.append(v_cur)
            p.append(p_cur)
        f_csv.writerow([project_id, "cor"] + v)
        f_csv.writerow([project_id, "p-value"] + p)




if __name__ == '__main__':
    labels = ['forks','contributor','commits','commit_comment',
        'req_opened','req_closed','req_merged','other','issue','issue_comment','watchers']
    rootpath = os.getcwd() + '\\data\\'
    start, end = 1, 3
    project_id = 961
    components_num = 10
    have_std = False
    projects_valid, data = Month_all_duration(rootpath, start, end, have_std = have_std)
    data_proc, scalers = process_data(data, start, end)

    ### 得到每个月的pca权重
    pca = get_pca(data_proc, scalers, components_num)
    ### scores记录使用pca方法计算得到的健康性
    scores = get_data_score(project_id, data_proc, projects_valid, scalers, pca, start, end)

    ### 得到每个月的信息熵方法的权重
    weights = get_weights(data_proc, scalers, start, end, have_std = have_std)
    ### scores_2记录使用信息熵方法得到的健康性
    scores_2 = entropy(project_id, projects_valid, data_proc, weights, scalers, start, end)

    ### compare 比较两种方法
    compare(project_id, start, end, scores, scores_2, filepath = 'Question_1')

    ### 
    # trend(project_id, data_proc, projects_valid, scores, start, end, filepath = 'Question_1',have_std = have_std)
    # for indicator in labels:
    #     trend_indicator(indicator, pca, start, end, have_std = have_std)
    # scores_all_proj = get_scores_projects(data_proc, projects_valid, scalers, pca, start, end)
    # health_hist(scores_all_proj, start, end)
    # t_test_data(projects_valid, data_proc, weights, scalers, pca, start, end)
