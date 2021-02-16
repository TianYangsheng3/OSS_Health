import pymysql
import csv
from datetime import datetime, date, timedelta
from requests_html import HTMLSession
import re

## connect to database
db = pymysql.connect(host='localhost', user='ystian', passwd='123456', db='ghtorrent_restore')
cursor = db.cursor()

project_id = 1088
# get created time
# id url owner_id name description
# language created_at forked_from deleted updated_at
sql_0 = "select * from projects where id = " + str(project_id)
cursor.execute(sql_0)
tmp = cursor.fetchone()
created_date = tmp[6].date()
last_date = created_date
base_url = tmp[1].replace('api.', '').replace('repos/', '')

# get all forks data since created_date
sql_forks = "select * from projects where forked_from=" + str(project_id) + " order by created_at "
cursor.execute(sql_forks)
data_forks = cursor.fetchall()
if data_forks:
    last_date_forks = data_forks[-1][6].date()
    if (last_date_forks - last_date).days > 0:
        last_date = last_date_forks
    print("get data forks")
else:
    data_forks = []

# get all commit data since created_date
# id    sha     author_id   committer_id    project_id
# created_at
sql_commit = "select * from commits where project_id =" + str(
    project_id) + " and created_at>='" + created_date.strftime('%Y-%m-%d') + "' order by created_at"
cursor.execute(sql_commit)
data_commits = cursor.fetchall()
if data_commits:
    last_date_commits = data_commits[-1][5].date()
    if (last_date_commits - last_date).days > 0:
        last_date = last_date_commits
    print("get data commits")
else:
    data_commits = []

# get all watchers data since created_date
# repo_id   user_id     created_at
sql_watcher = "select * from watchers where repo_id = " + str(project_id) + "  order by created_at"
cursor.execute(sql_watcher)
data_watcher = cursor.fetchall()
if data_watcher:
    last_date_watcher = data_watcher[-1][2].date()
    if (last_date_watcher - last_date).days > 0:
        last_date = last_date_watcher
    print("get data watcher")
else:
    data_watcher = []
# get all pull_request data since created_at
# id    head_repo_id    base_repo_id    head_commit_id  base_commit_id
# pullreq_id    intra_branch
# id    pull_request_id     created_at      action  actor_id
sql_pullreq = '''select * from (select * from pull_requests 
where pull_requests.base_repo_id = ''' + str(project_id) + ''') as tmp 
join pull_request_history on tmp.id = pull_request_history.pull_request_id 
 order by pull_request_history.created_at'''
cursor.execute(sql_pullreq)
data_pullreq = cursor.fetchall()
if data_pullreq:
    last_date_pullreq = data_pullreq[-1][9].date()
    if (last_date_pullreq - last_date).days > 0:
        last_date = last_date_pullreq
    print("get data pull_request")
else:
    data_pullreq = []

# get all issue data since created_at
# id    repo_id     reporter_id     assignee_id     pull_request
# pull_request_id   created_at  issue_id
sql_issue = "select * from issues where repo_id=" + str(project_id) + " order by created_at"
cursor.execute(sql_issue)
data_issue = cursor.fetchall()
if data_issue:
    last_date_issue = data_issue[-1][6].date()
    if (last_date_issue - last_date).days > 0:
        last_date = last_date_issue
    print("get data issue")
else:
    data_issue = []
# get all commit_comment data since created_at
# id    sha     author_id   committer_id    project_id
# created_at
# id    commit_id   user_id     body    line
# position      comment_id      created_at
sql_commit_comment = '''select * from (select * from 
commits where commits.project_id=''' + str(project_id) + ''') as tmp 
join commit_comments on tmp.id=commit_comments.commit_id 
ORDER BY commit_comments.created_at'''
cursor.execute(sql_commit_comment)
data_commit_comment = cursor.fetchall()
if data_commit_comment:
    last_date_commit_comment = data_commit_comment[-1][13].date()
    if (last_date_commit_comment - last_date).days > 0:
        last_date = last_date_commit_comment
    print("get data commit_comment")
else:
    data_commit_comment = []

# get all issue_comment since created_at
# id  repo_id  reporter_id  assignee_id  pull_request
# pull_request_id  created_at  issue_id
# issue_id  user_id     comment_id  created_at
sql_issue_comment = '''select * from (select * from issues where 
issues.repo_id=''' + str(project_id) + ''') as tmp join issue_comments on
tmp.id = issue_comments.issue_id order by issue_comments.created_at
'''
cursor.execute(sql_issue_comment)
data_issue_comment = cursor.fetchall()
if data_issue_comment:
    last_date_issue_comment = data_issue_comment[-1][11].date()
    if (last_date_issue_comment - last_date).days > 0:
        last_date = last_date_issue_comment
    print("get data issue_comment")
else:
    data_issue_comment = []

span = (last_date - created_date).days
headers = ['project_id', 'date', 'forks', 'committer_id',  'commits', 'commit_comment',
           'req_opened', 'req_closed', 'req_merged', 'other', 'issue', 'issue_comment', 'watchers']
# print(data_commits)
with open('getdata.csv', 'w') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)

    # project_id = 1088

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

db.close()
