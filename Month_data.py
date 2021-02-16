import numpy as np 
import os
import csv
import ast
from collections import namedtuple
from datetime import datetime, date, timedelta
from sklearn.preprocessing import normalize
import random

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
        if data_means[0]*len(commits)<0:      # 每月的commit数不能少于两个
            return False, 0 , 0
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
    for file in project_id:
        filepath = root_path + 'project_' + file + '.csv'
        flag,means_data, std_data = Month_one(filepath, i)
        if flag:
            projects_valid.append(int(file))
            data.append(list(means_data))
    return projects_valid


if __name__ == '__main__':
    data = []
    cluster_num = 4
    month_i = 5
    root_path = os.getcwd() + '/data/'
    projects_valid = Month_all(root_path, data, month_i)
    print(projects_valid)