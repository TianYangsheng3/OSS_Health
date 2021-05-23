Based on the GHTorrent dataset (mysql-2018-06-01, https://ghtorrent.org/downloads.html), this dataset is collected through extracting data from repos with a python theme and no less than 100 forks on GitHub. The data records the values ​​of the following indicators for each day from the creation date to 2018-06-01:

fork: the number of forks of the project per day
committer: the number of contributors to the project each day
commits: the number of commits per day of the project
commit_comment: The number of commit comments per day of the project
req_opened: The number of request open statuses of the project each day
req_closed: the number of request close statuses of the project each day
req_merged: The number of request merge states of the project each day
other: The number of request other statuses of the project each day
issue: the number of issues per day of the project
issue_comment: The number of comments the project issues per day
watchers: the number of viewers per day of the project

In the data folder, the projects.csv file records the id and creation date of all projects in the dataset. The other csv file naming rule is "project_projectid.csv", which records the daily data of the corresponding project. 

The dataset and the data collection method have been uploaded to GitHub, namely https://github.com/TianYangsheng3/OSS_Health.
