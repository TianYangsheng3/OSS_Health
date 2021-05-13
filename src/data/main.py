from Get_data import ProjEverydayData
from Get_data_mysql import Get_project
import pymysql, csv, os

#### 获得从start*100开始到end*100为止的项目的id和创建时间，即'project_id','created_at'
def ObtainProjects(start, end, FileRootPath):
    nums = (end-start)*100
    start = start*100
    
    ### host = '10.201.98.82'
    db = pymysql.connect(host='10.201.98.82', user='ystian', passwd='123456', db='ghtorrent_restore')
    cursor = db.cursor()

    url_projects = "select id, created_at from projects where projects.`language`='python' and projects.forked_from is NULL and deleted=0 order by id limit "
    projects_data = Get_project("projects", url_projects, cursor, start, nums)

    headers = ['project_id','created_at']
    
    # pro_filepath = FileRootPath + 'projects.csv'     ## linux path
    pro_filepath = FileRootPath + 'projects.csv'      ## wins path
    with open(pro_filepath,'w', newline="") as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(tuple(projects_data))

    db.close()

#### 获得从start*100开始到end*100为止的每个项目的数据，每隔100条输出一条日志
def ObtainData(start, end, FileRootPath):
    for i in range(start,end):
        begin = i*100
        nums = 100
        ProjEverydayData(begin, nums, FileRootPath)
        print("*************************")
        print("          "+str(i) + "        ")
        print("*************************")

if __name__ == '__main__':
    # FileRootpath = "data/"   ### linux path
    FileRootpath = "data\\"     ### wins path
    start, end = 200, 201
    ObtainData(start, end, FileRootpath)
    ObtainProjects(start, end, FileRootpath)

    
