from Get_data import fun
from Get_data_mysql import Get_project
import pymysql, csv, os
from cluster import k_means, Draw_graph, choose_project
from MyPCA import pca_1
from Month_data import Month_all

def tmp(start, nums):
    db = pymysql.connect(host='localhost', user='ystian', passwd='123456', db='ghtorrent_restore')
    cursor = db.cursor()

    url_projects = "select id, created_at from projects where projects.`language`='python' and projects.forked_from is NULL and deleted=0 order by id limit "
    # start, nums = 7900, 2100
    projects_data = Get_project("projects", url_projects, cursor, start, nums)

    headers = ['project_id','created_at']
    
    with open('data/projects.csv','w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(tuple(projects_data))

    db.close()


def Obtain_data(start, end):
    for i in range(start,end):
        start = i*100
        nums = 100
        fun(start, nums)
        print("*************************")
        print("          "+str(i) + "        ")
        print("*************************")
    # tmp(start*100, (end-start)*100)

def Process_data():
    # data = [[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]]
    data = []           # 要处理的数据，得到的是所有项目从创建开始第month_i个月的数据
    cluster_num = 4     # 聚类类别个数
    month_i = 20    
    choose_num = 3      # 从每类中选择的项目的个数    
    root_path = os.getcwd() + '/data/'        # 项目原始数据文件夹路径
    projects_valid = Month_all(root_path, data, month_i)    # 得到的有效的项目，project_id
    components_n = 11       # 主成分个数，当设置成‘mle’时暂时用不到
    if len(data)>1:
        kmeans_labels, kmeans_centers = k_means(data, cluster_num)
        Draw_graph(data, kmeans_labels, kmeans_centers, cluster_num, 0)     # 绘图每类类中心的条形图
        selects = choose_project(data, kmeans_labels, projects_valid, cluster_num, choose_num)      # 记录的是选出来的项目在projects_valid中的下标
        print("The index of selects is: ", selects)
        pca_1(data, components_n, selects, projects_valid)        # 运用主成分分析法得到各个指标的权重
    else:
        print("There is no project with a lifetime of more than " + str(month_i) +" months")

if __name__ == '__main__':
    start, end = 200, 220
    Obtain_data(start, end)
    tmp(start*100, (end-start)*100)
    # Process_data()
    
