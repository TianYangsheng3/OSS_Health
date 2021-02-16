import pymysql
import csv
from datetime import datetime, date, timedelta
from requests_html import HTMLSession
import re
from Get_data_mysql import Get, Get_project

'''
从mysql数据库中提取出我们想要的每个项目每一天的数据
'''

# 从mysql数据库中提取出我们想要的每个项目每一天的数据，例如从创建到现在的每一天的forks数量
def fun(start, nums):

    ## connect to database
    db = pymysql.connect(host='localhost', user='ystian', passwd='123456', db='ghtorrent_restore')
    cursor = db.cursor()

    # get projects
    url_projects = "select id, created_at from projects where projects.`language`='python' and projects.forked_from is NULL and deleted=0 order by id limit "
    #start, nums = 0, 5
    projects_data = Get_project("projects", url_projects, cursor, start, nums)

    for row in range(len(projects_data)):
        project_id = projects_data[row][0]
        created_date = projects_data[row][1].date()
        last_date = created_date

        # get forks
        sql_forks = "select * from projects where forked_from=" + str(project_id) + " order by created_at "
        last_date, data_forks = Get(project_id, "forks", sql_forks, cursor, 6, last_date)

        # get commit
        sql_commit = "select * from commits where project_id =" + str(project_id) + " and created_at>='" + created_date.strftime('%Y-%m-%d') + "' order by created_at"
        last_date, data_commits = Get(project_id,"commits", sql_commit, cursor, 5, last_date)

        # get watchers
        sql_watcher = "select * from watchers where repo_id = " + str(project_id) + "  order by created_at"
        last_date, data_watcher = Get(project_id, "watchers", sql_watcher, cursor, 2, last_date)

        # get pull_request
        sql_pullreq = '''select * from (select * from pull_requests 
        where pull_requests.base_repo_id = ''' + str(project_id) + ''') as tmp 
        join pull_request_history on tmp.id = pull_request_history.pull_request_id 
        order by pull_request_history.created_at'''
        last_date, data_pullreq = Get(project_id, "pull_request", sql_pullreq, cursor, 9, last_date)

        # get issue
        sql_issue = "select * from issues where repo_id=" + str(project_id) + " order by created_at"
        last_date, data_issue = Get(project_id,"issues", sql_issue, cursor, 6, last_date)

        # get commit_comment
        sql_commit_comment = '''select * from (select * from 
        commits where commits.project_id=''' + str(project_id) + ''') as tmp 
        join commit_comments on tmp.id=commit_comments.commit_id 
        ORDER BY commit_comments.created_at'''   
        last_date, data_commit_comment = Get(project_id,"commit_comment", sql_commit_comment, cursor, 13, last_date)     

        # get issue_comment
        sql_issue_comment = '''select * from (select * from issues where 
        issues.repo_id=''' + str(project_id) + ''') as tmp join issue_comments on
        tmp.id = issue_comments.issue_id order by issue_comments.created_at
        '''  
        last_date,  data_issue_comment = Get(project_id,"issue_comment", sql_issue_comment, cursor, 11, last_date)     


        span = (last_date - created_date).days
        headers = ['project_id', 'date', 'forks', 'committer_id',  'commits', 'commit_comment',
                'req_opened', 'req_closed', 'req_merged', 'other', 'issue', 'issue_comment', 'watchers']

        # write to file
        file_path = "data/project_" + str(project_id) + ".csv"
        with open(file_path, 'w') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(headers)

            length_fork = len(data_forks)
            cur_row_fork = 0

            length_commit = len(data_commits)
            cur_row_commit = 0

            length_watcher = len(data_watcher)
            cur_row_watcher = 0

            length_pullreq = len(data_pullreq)
            cur_row_pullreq = 0

            length_issue = len(data_issue)
            cur_row_issue = 0

            length_commit_comment = len(data_commit_comment)
            cur_row_commit_comment = 0

            length_issue_comment = len(data_issue_comment)
            cur_row_issue_comment = 0

            for i in range(span + 1):
                today = created_date + timedelta(days=i)
                # get forks
                forks = 0
                while cur_row_fork < length_fork and (data_forks[cur_row_fork][6].date() - today).days <= 0:
                    if (data_forks[cur_row_fork][6].date() - today).days == 0:
                        forks += 1
                    cur_row_fork += 1
                #   get committer
                committer = {}
                commits = 0
                while cur_row_commit < length_commit and (data_commits[cur_row_commit][5].date() - today).days <= 0:
                    if (data_commits[cur_row_commit][5].date() - today).days == 0:
                        commits += 1
                        person_id = data_commits[cur_row_commit][3]
                        if person_id in committer:
                            committer[person_id] += 1
                        else:
                            committer[person_id] = 1
                    cur_row_commit += 1

                # get commit comment
                commit_comment = 0
                while cur_row_commit_comment < length_commit_comment and (
                        data_commit_comment[cur_row_commit_comment][13].date() - today).days <= 0:
                    if (data_commit_comment[cur_row_commit_comment][13].date() - today).days == 0:
                        commit_comment += 1
                    cur_row_commit_comment += 1

                # get pull request
                pull_request_opened = 0
                pull_request_closed = 0
                pull_request_merged = 0
                pull_request_others = 0
                while cur_row_pullreq < length_pullreq and (data_pullreq[cur_row_pullreq][9].date() - today).days <= 0:
                    if (data_pullreq[cur_row_pullreq][9].date() - today).days == 0:
                        if data_pullreq[cur_row_pullreq][10] == 'opened':
                            pull_request_opened += 1
                        elif data_pullreq[cur_row_pullreq][10] == 'closed':
                            pull_request_closed += 1
                        elif data_pullreq[cur_row_pullreq][10] == 'merged':
                            pull_request_merged += 1
                        else:
                            pull_request_others += 1
                    cur_row_pullreq += 1

                # get issue
                issue = 0
                while cur_row_issue < length_issue and (data_issue[cur_row_issue][6].date() - today).days <= 0:
                    if (data_issue[cur_row_issue][6].date() - today).days == 0:
                        issue += 1
                    cur_row_issue += 1

                # get issue comment
                issue_comment = 0
                while cur_row_issue_comment < length_issue_comment and (
                        data_issue_comment[cur_row_issue_comment][11].date() - today).days <= 0:
                    if (data_issue_comment[cur_row_issue_comment][11].date() - today).days == 0:
                        issue_comment += 1
                    cur_row_issue_comment += 1

                # get watchers
                watcher = 0
                while cur_row_watcher < length_watcher and (data_watcher[cur_row_watcher][2].date() - today).days <= 0:
                    if (data_watcher[cur_row_watcher][2].date() - today).days == 0:
                        watcher += 1
                    cur_row_watcher += 1
                # headers = ['project_id', 'date', 'forks', 'committer_id', 'commits', 'commit_comment',
                #           'req_opened', 'req_closed', 'req_merged', 'other', 'issue', 'issue_comment', 'watchers']
                one_data = (project_id, today, forks, committer, commits, commit_comment, pull_request_opened,
                            pull_request_closed, pull_request_merged, pull_request_others, issue, issue_comment, watcher)
                f_csv.writerow(one_data)
            print("################ get " + str(row+1) + " data "+str(project_id)+" ################ ")
    db.close()
