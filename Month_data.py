import numpy as np 
import os
import csv
import ast
from collections import namedtuple
from datetime import datetime, date, timedelta
from sklearn.preprocessing import normalize
import random
import pandas as pd
from sklearn.preprocessing import normalize,MinMaxScaler,Normalizer

'''
得到每个项目的评价指标的月平均数据和标准差
'''


# 将字符串解析成date
def parse_date(s):
    year_s, month_s, day_s = s.split('-')
    return date(int(year_s), int(month_s), int(day_s))


# 得到某个项目filepath的第i月的平均数据和标准差
def Month_one(filepath, i):
    with open(filepath, 'r') as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)

        Row = namedtuple('Row', headers)
        flag = True
        created_date = None
        # data = [[] for i in range(11)]
        forks,committer_id,commits,commit_comment = [], [], [], []
        req_opened,req_closed,req_merged,other,issue,issue_comment,watchers = [], [], [], [], [], [], []
        # project_id,date,forks,committer_id,commits,commit_comment,req_opened,req_closed,req_merged,other,issue,issue_comment,watchers
        for r in  f_csv:
            row = Row(*r)
            cur = parse_date(row.date)
            if flag:
                flag = False
                created_date = cur
            dist = (cur.month-created_date.month) + (cur.year - created_date.year)*12
            if dist == i:
                forks.append(int(row.forks))
                commits.append(int(row.commits))
                commit_comment.append(int(row.commit_comment))
                req_opened.append(int(row.req_opened))
                req_closed.append(int(row.req_closed))
                req_merged.append(int(row.req_merged))
                other.append(int(row.other))
                issue.append(int(row.issue))
                issue_comment.append(int(row.issue_comment))
                watchers.append(int(row.watchers))
                committer_dict = ast.literal_eval(row.committer_id)
                committer_id.append(committer_dict)
            elif dist>i:
                break
            else:
                continue
        if len(forks)<1:
            return False, 0, 0
        developers = set()
        for dic in committer_id:
            for k, v in dic.items():
                if k not in developers:
                    developers.add(k)
        # print(developers)
        developer_num = [len(developers)]*len(forks)

        data = np.array([forks,developer_num,commits,commit_comment,req_opened,req_closed,req_merged,other,issue,issue_comment,watchers])
        # print(data)
        data_means = np.mean(data, axis=1)
        data_std = np.std(data, axis=1)
        # if data_means[2]*len(commits)<0:      # 每月的commit数不能少于两个
        #     return False, data_means, data_std
        return True, data_means, data_std

# 得到某个项目filepath的【start，end）月的平均数据和标准差
def Month_duration(root_path, project_id, start, end):
    filepath = root_path  + 'project_' + str(project_id) + '.csv'
    with open(filepath, 'r') as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)

        Row = namedtuple('Row', headers)
        flag = True
        created_date = None
        # data = [[] for i in range(11)]
        forks,committer_id,commits= [[] for i in range(end-start)], [[] for i in range(end-start)], [[] for i in range(end-start)]
        commit_comment,req_opened,req_closed = [[] for i in range(end-start)],[[] for i in range(end-start)],[[] for i in range(end-start)]
        req_merged,other,issue = [[] for i in range(end-start)], [[] for i in range(end-start)], [[] for i in range(end-start)]
        issue_comment,watchers = [[] for i in range(end-start)], [[] for i in range(end-start)]
        for r in  f_csv:
            row = Row(*r)
            cur = parse_date(row.date)
            if flag:
                flag = False
                created_date = cur
            dist = (cur.month-created_date.month) + (cur.year - created_date.year)*12

            # project_id,date,forks,committer_id,commits,commit_comment,req_opened,req_closed,req_merged,other,issue,issue_comment,watchers
            if dist>=start and dist<end:
                forks[dist-start].append(int(row.forks))
                commits[dist-start].append(int(row.commits))
                commit_comment[dist-start].append(int(row.commit_comment))
                req_opened[dist-start].append(int(row.req_opened))
                req_closed[dist-start].append(int(row.req_closed))
                req_merged[dist-start].append(int(row.req_merged))
                other[dist-start].append(int(row.other))
                issue[dist-start].append(int(row.issue))
                issue_comment[dist-start].append(int(row.issue_comment))
                watchers[dist-start].append(int(row.watchers))
                committer_dict = ast.literal_eval(row.committer_id)
                committer_id[dist-start].append(committer_dict)
            elif dist>=end:
                break
            else:
                continue
        
        if len(forks[-1])<1:
            return False, 0, 0

        data_means = []
        data_std = []
        for i in range(end-start):
            developers = set()
            for dic in committer_id[i]:
                for k, v in dic.items():
                    if k not in developers:
                        developers.add(k)
            developer_num = [len(developers)]*len(forks[i])

            cur_data = np.array([forks[i],developer_num,commits[i],commit_comment[i],req_opened[i],req_closed[i],req_merged[i],
                    other[i],issue[i],issue_comment[i],watchers[i]])

            cur_data_means = np.mean(cur_data, axis=1)
            cur_data_std = np.std(cur_data, axis=1)

            data_means.append(cur_data_means)
            data_std.append(cur_data_std)

        return True, data_means, data_std
                     
# 得到root_path路径下所有文件（即项目）从创建起第i个月的数据，存在data中
def Month_all(root_path, data, i):
    project_id = []
    projects_path = root_path + 'projects.csv'

    # 从projects.csv文件中得到所有文件名
    with open(projects_path, 'r') as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)

        Row = namedtuple('Row',headers)     # 使用字段名访问数据(步骤1)
        # print(headers, Row)
        for r in f_csv:
            row = Row(*r)   # 使用字段名访问数据（步骤2）
            project_id.append(row.project_id)

    # 得到所有文件第i月的数据
    projects_valid = [] #有效的project id
    # count = 0
    for file in project_id:
        filepath = root_path + 'project_' + file + '.csv'
        # project_id,date,forks,committer_id,commits,commit_comment,req_opened,req_closed,req_merged,other,issue,issue_comment,watchers
        if is_valid(filepath, 'forks', 100) and is_valid(filepath, 'commits', 100):
            # count += 1
            flag,means_data, std_data = Month_one(filepath, i)
            if flag:
                projects_valid.append(int(file))
                data.append(means_data)       # 只选择均值作为特征
                # data.append(list(means_data) + list(std_data))  # 将标准差和均值都作为特征
    # print(count)
    return projects_valid


# 得到root_path路径下所有文件（即项目）从创建起第i个月的数据，存在data中
def Month_all_duration(root_path, start, end):
    project_id = []
    data = []
    projects_path = root_path + 'projects.csv'

    # 从projects.csv文件中得到所有文件名
    with open(projects_path, 'r') as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)

        Row = namedtuple('Row',headers)     # 使用字段名访问数据(步骤1)
        # print(headers, Row)
        for r in f_csv:
            row = Row(*r)   # 使用字段名访问数据（步骤2）
            project_id.append(row.project_id)

    # 得到所有文件第i月的数据
    projects_valid = [] #有效的project id
    # count = 0
    for file in project_id:
        filepath = root_path + 'project_' + file + '.csv'
        # project_id,date,forks,committer_id,commits,commit_comment,req_opened,req_closed,req_merged,other,issue,issue_comment,watchers
        if is_valid(filepath, 'forks', 100) and is_valid(filepath, 'commits', 100):
            # count += 1
            flag, means_data, std_data = Month_duration(root_path, int(file), start, end)
            if flag:
                projects_valid.append(int(file))
                data.append(means_data)       # 只选择均值作为特征
                # data.append(list(means_data) + list(std_data))  # 将标准差和均值都作为特征
    # print(count)
    return projects_valid, data

# 将数据整理成每月汇总数据
def process_data(data, start, end):
    data_proc = [[] for k in range(end-start)]      # 从start到end，每个月的所有项目数据都放在一个列表中
    scaler = []
    for i in range(len(data)):
        for j in range(end-start):
            data_proc[j].append(data[i][j])
    data_proc = np.array(data_proc)
    for k in range(end-start):
        minmax_scaler = MinMaxScaler(copy=False)
        minmax_scaler.fit(data_proc[k])
        scaler.append(minmax_scaler)
        # normal = Normalizer().fit(data_proc[k])
        # scaler.append(normal)
        # data_proc[k] = normalize(data_proc[k], axis=0)
    return data_proc, scaler



# 选出feature > bound 的项目
def is_valid(filepath, features, bound):
    data = pd.read_csv(filepath)
    if data[features].sum()<bound:
        return False
    else:
        return True

if __name__ == '__main__':
    # data = []
    root_path = os.getcwd() + '\\data\\'
    project_id = 1486
    projects_valid, data = Month_all_duration(root_path, 0,3)
    # flag, data_means, data_std = Month_duration(root_path, project_id, 0, 3)
    # print(projects_valid)
    print(type(data))
    data_proc, scaler = process_data(data, 0 ,3)
    print(type(data_proc))
    print(data[0])
    print(data_proc[0][0])
    print(scaler[0].transform([data_proc[0][0]]))
    print(len(data_proc))