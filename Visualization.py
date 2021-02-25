import numpy as np
import matplotlib.pyplot as plt
from Month_data import Month_all, Month_one
from cluster import k_means, choose_project
from MyPCA import pca_1, compute_newx, LinearReg
import os
from sklearn.preprocessing import normalize
import csv
import math

'''
画出类中心、项目的健康性、特征月变化折线图
'''


# 画出每个月的类中心健康性变化、特征变化的折线图
def view_cluster_centers(kmeans_labels, kmeans_centers, scores_all,cluster_num, months):
    x = np.array(range(months))
    index_all = []
    # print("scores_all:",scores_all)
    for i in range(months):
        index = np.argsort(np.array(scores_all[i]))     # 评分从低到高
        index_all.append(index)

    for cluster_i in range(cluster_num):
        health = []
        forks,developer_num,commits,commit_comment,req_opened = [], [], [], [], []
        req_closed,req_merged,other,issue,issue_comment,watchers = [], [], [], [], [], []
        for month_i in range(months):
            cur = index_all[month_i][cluster_i]     #按照评分从低到高，排cluster_i的类中心在第month_i月的索引
            # print("cluster "+ str(cluster_i)+ " index: " + str(cur))
            health.append(scores_all[month_i][cur])
            forks.append(kmeans_centers[month_i][cur][0])
            developer_num.append(kmeans_centers[month_i][cur][1])
            commits.append(kmeans_centers[month_i][cur][2])
            # commit_comment.append(kmeans_centers[month_i][cur][3])
            # req_opened.append(kmeans_centers[month_i][cur][4])
            # req_closed.append(kmeans_centers[month_i][cur][5])
            # req_merged.append(kmeans_centers[month_i][cur][6])
            # other.append(kmeans_centers[month_i][cur][7])
            # issue.append(kmeans_centers[month_i][cur][8])
            # issue_comment.append(kmeans_centers[month_i][cur][9])
            watchers.append(kmeans_centers[month_i][cur][10])
        plt.plot(x, health, label = 'health')
        plt.plot(x, forks, label = 'forks')
        plt.plot(x, developer_num, label = 'developer_num')
        plt.plot(x, commits, label = 'commits')
        # plt.plot(x, commit_comment, label = 'commit_comment')
        # plt.plot(x, req_opened, label = 'req_opened')
        # plt.plot(x, req_closed, label = 'req_closed')
        # plt.plot(x, req_merged, label = 'req_merged')
        # plt.plot(x, other, label = 'other')
        # plt.plot(x, issue, label = 'issue')
        # plt.plot(x, issue_comment, label = 'issue_comment')
        plt.plot(x, watchers, label = 'watchers')
        plt.xlabel('month')
        plt.ylabel('value')
        plt.title('development of cluster center: ' + str(cluster_i) + ' during ' + str(months) + ' months' )
        plt.legend()
        plt.show()


def view_cluster_centers_2(kmeans_labels, kmeans_centers, scores_all,cluster_num, months):
    x = np.array(range(months))
    index_all = []
    # print("scores_all:",scores_all)
    for i in range(months):
        index = np.argsort(np.array(scores_all[i]))     # 评分从低到高
        index_all.append(index)

    for cluster_i in range(cluster_num):
        
        labels = ['forks','committer','commits','commit_comment',
        'req_opened','req_closed','req_merged','other','issue','issue_comment','watchers']
        forks,developer_num,commits,commit_comment,req_opened = [], [], [], [], []
        req_closed,req_merged,other,issue,issue_comment,watchers = [], [], [], [], [], []
        for i in range(len(labels)):
            health = []
            means = []
            std = []
            for month_i in range(months):
                cur = index_all[month_i][cluster_i]     #按照评分从低到高，排cluster_i的类中心在第month_i月的索引
                # print("cluster "+ str(cluster_i)+ " index: " + str(cur))
                health.append(scores_all[month_i][cur])
                means.append(kmeans_centers[month_i][cur][i])
                std.append(kmeans_centers[month_i][cur][i+11])
                # forks.append(kmeans_centers[month_i][cur][0])
                # developer_num.append(kmeans_centers[month_i][cur][1])
                # commits.append(kmeans_centers[month_i][cur][2])
                # commit_comment.append(kmeans_centers[month_i][cur][3])
                # req_opened.append(kmeans_centers[month_i][cur][4])
                # req_closed.append(kmeans_centers[month_i][cur][5])
                # req_merged.append(kmeans_centers[month_i][cur][6])
                # other.append(kmeans_centers[month_i][cur][7])
                # issue.append(kmeans_centers[month_i][cur][8])
                # issue_comment.append(kmeans_centers[month_i][cur][9])
                # watchers.append(kmeans_centers[month_i][cur][10])
            plt.plot(x, health, label = 'health')
            plt.plot(x, means, label = labels[i]+'_mean')
            plt.plot(x, std, label = labels[i]+'_std')
            # plt.plot(x, forks, label = 'forks')
            # plt.plot(x, developer_num, label = 'developer_num')
            # plt.plot(x, commits, label = 'commits')
            # plt.plot(x, commit_comment, label = 'commit_comment')
            # plt.plot(x, req_opened, label = 'req_opened')
            # plt.plot(x, req_closed, label = 'req_closed')
            # plt.plot(x, req_merged, label = 'req_merged')
            # plt.plot(x, other, label = 'other')
            # plt.plot(x, issue, label = 'issue')
            # plt.plot(x, issue_comment, label = 'issue_comment')
            # plt.plot(x, watchers, label = 'watchers')
            plt.xlabel('month')
            plt.ylabel('value')
            plt.title('development of cluster center: ' + str(cluster_i) + ' during ' + str(months) + ' months' )
            plt.legend()
            plt.show()
            

def pre_view(root_path, months,cluster_num,components_n, selects):    # months个月的变化趋势，聚类类别个数cluster_num， pca的主成分个数
    # 画类中心变化趋势所需要的数据
    kmeans_labels_all = []
    kmeans_centers_all = []
    scores_all = []   
    # 画项目变化趋势需要的数据
    data_projects_all = []
    cluster_order_all = []
    scores_projects_all = []  

    # root_path = os.getcwd() + '\\data\\' 

    for month_i in range(months):
        data = []           # 要处理的数据，得到的是所有项目从创建开始第month_i个月的数据
        projects_valid = Month_all(root_path, data, month_i)    # 得到的有效的项目，project_id

        if len(data)>1:
            data = normalize(X = data, axis=0)
            # print("(pre_view) data is ", data)
            # kmeans_labels, kmeans_centers = k_means(data, cluster_num)
            # forks,developer_num,commits,commit_comment,req_opened,req_closed,req_merged,other,issue,issue_comment,watchers  
            pca = pca_1(data, components_n)        # 运用主成分分析法得到各个指标的权重
            # scores_centers = compute_newx(pca, kmeans_centers)

            # kmeans_labels_all.append(kmeans_labels)
            # kmeans_centers_all.append(kmeans_centers)
            # scores_all.append(scores_centers)

            cur_data = []
            cur_scores = []
            # scores_centers_sorted = sorted(scores_centers)
            # cur_order = [scores_centers_sorted.index(i) for i in scores_centers]
            for projects in selects:
                # tmp_index = []
                tmp_data = []
                for p_id in projects:
                    tmp_index=projects_valid.index(p_id)
                    tmp_data.append(data[tmp_index])
                cur_data.append(tmp_data)
                tmp_scores = compute_newx(pca, tmp_data)
                cur_scores.append(tmp_scores)
            data_projects_all.append(cur_data)
            scores_projects_all.append(cur_scores)
            # cluster_order_all.append(cur_order)
        else:
            print("(pre_view) There is no project with a lifetime of more than " + str(month_i) +" months")
            break
    # return kmeans_labels_all, kmeans_centers_all, scores_all, data_projects_all, scores_projects_all, cluster_order_all
    return data_projects_all, scores_projects_all



def view_project(selects, data, scores, cluters, cluster_num, months, ind = 0):
    x = np.array(range(months))
    print("scores：",scores)
    for cluster_i in range(cluster_num):
        health = []
        order = []
        forks,developer_num,commits,commit_comment,req_opened = [], [], [], [], []
        req_closed,req_merged,other,issue,issue_comment,watchers = [], [], [], [], [], []
        for month_i in range(months):
            # cur = index_all[month_i][cluster_i]     #按照评分从低到高，排cluster_i的类中心在第month_i月的索引
            # print("cluster "+ str(cluster_i)+ " index: " + str(cur))
            health.append(scores[month_i][cluster_i][ind])
            order.append(cluters[month_i][cluster_i]/20)
            # forks.append(data[month_i][cluster_i][ind][0])
            # developer_num.append(data[month_i][cluster_i][ind][1])
            # commits.append(data[month_i][cluster_i][ind][2])
            # commit_comment.append(data[month_i][cluster_i][ind][3])
            # req_opened.append(data[month_i][cluster_i][ind][4])
            # req_closed.append(data[month_i][cluster_i][ind][5])
            # req_merged.append(data[month_i][cluster_i][ind][6])
            # other.append(data[month_i][cluster_i][ind][7])
            # issue.append(data[month_i][cluster_i][ind][8])
            # issue_comment.append(data[month_i][cluster_i][ind][9])
            # watchers.append(data[month_i][cluster_i][ind][10])
        plt.plot(x, health, label = 'health')
        plt.plot(x, order, label = 'order')
        # plt.plot(x, forks, label = 'forks')
        # plt.plot(x, developer_num, label = 'developer_num')
        # plt.plot(x, commits, label = 'commits')
        # plt.plot(x, commit_comment, label = 'commit_comment')
        # plt.plot(x, req_opened, label = 'req_opened')
        # plt.plot(x, req_closed, label = 'req_closed')
        # plt.plot(x, req_merged, label = 'req_merged')
        # plt.plot(x, other, label = 'other')
        # plt.plot(x, issue, label = 'issue')
        # plt.plot(x, issue_comment, label = 'issue_comment')
        # plt.plot(x, watchers, label = 'watchers')
        plt.xlabel('month')
        plt.ylabel('value')
        plt.title('development of project_id: ' + str(selects[cluster_i][ind]) + ' during ' + str(months) + ' months' )
        plt.legend()
        plt.show()    


def view_project_2(selects, data, scores, cluters, cluster_num, months, ind = 0):
    x = np.array(range(months))
    print("scores：",scores)
    print("data: ", data)
    labels = ['forks','committer','commits','commit_comment',
        'req_opened','req_closed','req_merged','other','issue','issue_comment','watchers']
    for cluster_i in range(cluster_num):

        for i in range(len(labels)):
            health = []
            order = []
            means = []
            std = []
            forks,developer_num,commits,commit_comment,req_opened = [], [], [], [], []
            req_closed,req_merged,other,issue,issue_comment,watchers = [], [], [], [], [], []
            for month_i in range(months):
                # cur = index_all[month_i][cluster_i]     #按照评分从低到高，排cluster_i的类中心在第month_i月的索引
                # print("cluster "+ str(cluster_i)+ " index: " + str(cur))
                health.append(scores[month_i][cluster_i][ind])
                # order.append(cluters[month_i][cluster_i]/cluster_num)
                means.append(data[month_i][cluster_i][ind][i])
                # std.append(data[month_i][cluster_i][ind][i+11])
                # forks.append(data[month_i][cluster_i][ind][0])
                # developer_num.append(data[month_i][cluster_i][ind][1])
                # commits.append(data[month_i][cluster_i][ind][2])
                # commit_comment.append(data[month_i][cluster_i][ind][3])
                # req_opened.append(data[month_i][cluster_i][ind][4])
                # req_closed.append(data[month_i][cluster_i][ind][5])
                # req_merged.append(data[month_i][cluster_i][ind][6])
                # other.append(data[month_i][cluster_i][ind][7])
                # issue.append(data[month_i][cluster_i][ind][8])
                # issue_comment.append(data[month_i][cluster_i][ind][9])
                # watchers.append(data[month_i][cluster_i][ind][10])
            plt.plot(x, health, label = 'health')
            # plt.plot(x, order, label = 'order')
            plt.plot(x, means, label = labels[i]+'_mean')
            # plt.plot(x, std, label = labels[i]+'_std')
            # plt.plot(x, forks, label = 'forks')
            # plt.plot(x, developer_num, label = 'developer_num')
            # plt.plot(x, commits, label = 'commits')
            # plt.plot(x, commit_comment, label = 'commit_comment')
            # plt.plot(x, req_opened, label = 'req_opened')
            # plt.plot(x, req_closed, label = 'req_closed')
            # plt.plot(x, req_merged, label = 'req_merged')
            # plt.plot(x, other, label = 'other')
            # plt.plot(x, issue, label = 'issue')
            # plt.plot(x, issue_comment, label = 'issue_comment')
            # plt.plot(x, watchers, label = 'watchers')
            plt.xlabel('month')
            plt.ylabel('value')
            plt.title('development of project_id: ' + str(selects[cluster_i][ind]) + ' during ' + str(months) + ' months' )
            plt.legend()
            plt.show()   

def pre_choose_porject_id(root_path, months, cluster_num, choose_num = 1):
    res = []
    data = []           # 要处理的数据，得到的是所有项目从创建开始第month_i个月的数据
    # choose_num = 1      # 从每类中选择的项目的个数    
    # root_path = os.getcwd() + '\\data\\'        # 项目原始数据文件夹路径
    projects_valid = Month_all(root_path, data, months-1)    # 得到的有效的项目，project_id
    # components_n = 11       # 主成分个数，当设置成‘mle’时暂时用不到
    if len(data)>1:
        kmeans_labels, kmeans_centers = k_means(data, cluster_num)
        selects = choose_project(data, kmeans_labels, projects_valid, cluster_num, choose_num)      # 记录的是选出来的项目在projects_valid中的下标
        for porjects in selects:
            tmp = []
            for v in porjects:
                tmp.append(projects_valid[v])
            res.append(tmp)
        return res
    else:
        print("(pre_choose_porject_id) There is no project with a lifetime of more than " + str(months) +" months")
        return None


# 得到第month个月的健康性与分类标记
def health_cluster(root_path, month, num_components,cluster_num):
    data = []
    projects_valid = Month_all(root_path, data, month)

    if len(data)>1:
        kmeans_labels, kmeans_centers = k_means(data, cluster_num)
        pca = pca_1(data, num_components)
        scores = compute_newx(pca, data, True)
        # LinearReg(data, scores)
        header = ['project_id', 'cluster', 'health']
        to_file = 'month_'+str(month)+'.csv'
        with open(to_file, 'w') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(header)
            for i in range(len(projects_valid)):
                if scores[i]>0:
                    f_csv.writerow((projects_valid[i], kmeans_labels[i], math.log(scores[i])))
    else:
        print("(health_cluster) There is no project with a lifetime of more than " + str(months) +" months")



# 得到第month个月的健康性与分类标记
def health_dist(root_path, month, num_components):
    data = []
    projects_valid = Month_all(root_path, data, month)
    _projects_id = []
    _scores = []
    # print("Length of projects_valid: ", len(projects_valid))
    # print(projects_valid)
    if len(data)>1:
        pca = pca_1(data, num_components)
        scores = compute_newx(pca, data, True)
        # LinearReg(data, scores)
        header = ['project_id', 'health']
        to_file = 'month_'+str(month)+'.csv'
        with open(to_file, 'w', newline="") as f:
            f_csv = csv.writer(f)
            f_csv.writerow(header)
            for i in range(len(projects_valid)):
                if scores[i]>=0:
                    f_csv.writerow((projects_valid[i], scores[i]))
                    _projects_id.append(projects_valid[i])
                    _scores.append(scores[i])

        print("Length of projects_valid: ", len(_projects_id))
        print(_projects_id)
        fig, ax = plt.subplots()
        num_bins = 50

        # the histogram of the data
        n, bins, patches = ax.hist(_scores, num_bins, density=True)

        ax.set_xlabel('scores')
        ax.set_ylabel('Probability density')
        ax.set_title('scores of 20,000 porjects')

        # Tweak spacing to prevent clipping of ylabel
        fig.tight_layout()
        plt.show()


    else:
        print("(health_cluster) There is no project with a lifetime of more than " + str(months) +" months")


def trend(project_id, data, score, month, pca):
    pass



if __name__ == '__main__':
    months,cluster_num,components_n,choose_num = 20, 11, 11, 1
    root_path = os.getcwd() + '\\data\\'
    # selects = pre_choose_porject_id(root_path, months, cluster_num, choose_num)
    selects = [[1486], [1799], [14901], [3894],[4023], [772], [367], [1891],[2160],[1142],[1729]]
    # print("(pre_choose_porject_id) selects is ",selects)
    # kmeans_labels_all, kmeans_centers_all, scores_all,  data_projects_all, scores_projects_all, cluster_order_all = pre_view(root_path,months, cluster_num, components_n, selects)
    # view_cluster_centers_2(kmeans_labels_all, kmeans_centers_all, scores_all, cluster_num, months)
    data_projects_all, scores_projects_all = pre_view(root_path,months, cluster_num, components_n, selects)
    view_project_2(selects, data_projects_all, scores_projects_all, 0, cluster_num, months)
    # health_cluster(root_path, months, components_n,cluster_num)
    # health_dist(root_path, months, components_n)





