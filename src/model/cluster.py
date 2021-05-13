import numpy as np 
from sklearn.cluster import KMeans 
from datetime import datetime, date, timedelta
from sklearn.preprocessing import normalize
import random,os
import matplotlib
import matplotlib.pyplot as plt 

'''
采用kmeans聚类，按照评价指标对项目聚类
'''        


def k_means(data, k):
    # 聚类
    # 首先得到每个项目每个月的特征向量，例如 P_i = (x1,x2,...,x8)代表项目P在从创建开始（在GitHub上的created_at）第i个月的各个特征的值
    X = data[:]
    X = normalize(X = X, axis=0)
    X = np.array(X)
    kmeans = KMeans(n_clusters=k, init='random', random_state=0, max_iter=500).fit(X)
    return list(kmeans.labels_), list(kmeans.cluster_centers_)
    
# 画出每类类中心的各个评价指标对比的条形图
def Draw_graph(data, x_labels, centers, cluster_num, n = 0):
    # labels = ['forks','committer','commits','commit_comment',
    # 'req_opened','req_closed','req_merged','other','issue','issue_comment','watchers']
    labels = ['forks','committer','commits','commit_comment',
    'req_opened','req_closed','req_merged','other','issue','issue_comment','watchers',
    'forks_std','committer_std','commits_std','commit_comment_std',
    'req_opened_std','req_closed_std','req_merged_std','other_std','issue_std','issue_comment_std','watchers_std']   
    x = np.arange(len(labels))  # the label locations
    width = 0.15 if cluster_num<5 else 0.15*5/cluster_num       # the width of the bars

    fig, ax = plt.subplots()
    rects = []
    for i in range(cluster_num):
        pos = x-cluster_num/2*width + (2*i+1)*width/2
        rect = ax.bar(pos, centers[i], width, label = str(i))
        rects.append(rect)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('feature_value')
    ax.set_title(str(cluster_num)+' cluters')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()


    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    # for rect in rects:
    #     autolabel(rect)

    fig.tight_layout()

    plt.show()    

## 从各类项目（project_label）中随机选择n个项目出来进行观察
def choose_project(data, project_label, projects_valid, cluster_num, n):
    projects = [ [] for i in range(cluster_num)]
    selects = []    # 记录的是选出来的项目在projects_valid中的下标
    for i in range(len(project_label)):
        cur_cluster = project_label[i]
        projects[cur_cluster].append(i)

    for i in range(cluster_num):
        print("cluster " + str(i) + " count is " + str(len(projects[i])) )       #输出每类的项目个数
        if len(projects[i])>n:
            tmp = random.sample(projects[i], n)
            selects.append(tmp)
        else:
            selects.append(projects[i])
            print("Error: cluster " + str(i) + " is not enough")
    for i in range(cluster_num):
        print("********************* The cluster " + str(i) + " ********************* ")
        for j in selects[i]:
            print([round(v, 4) for v in data[j]])      # 选出来的每类的标准化后的数据
    return selects


# if __name__ == '__main__':
#     data = []
#     cluster_num = 5
#     root_path = os.getcwd() + '\\data\\'

#     projects_valid = Month_all(root_path, data, 5)

#     if len(data)>1:
#         kmeans_labels, kmeans_centers = k_means(data, cluster_num)
#         Draw_graph(data, kmeans_labels, kmeans_centers, cluster_num, 0)