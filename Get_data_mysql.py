import pymysql
from datetime import datetime, date, timedelta

'''
通用函数：从数据库中取出相关数据
'''

# 得到项目从start开始的nums条项目数据['project_id','created_at']
def Get_project(item, sql, cursor, start, nums):
    sql = sql + str(start) + ","+ str(nums)
    cursor.execute(sql)
    res_data = cursor.fetchall()
    print("get data "+item)
    return res_data


# 得到项目project_id的item（例如forks）所有数据
def Get(project_id, item, sql, cursor, date_col,last_date):
    cursor.execute(sql)
    res_data = cursor.fetchall()
    if res_data:
        last_date_cur = res_data[-1][date_col].date()
        if (last_date_cur - last_date).days > 0:
            last_date = last_date_cur
        #print("get data "+ item +", " + str(project_id))
        return last_date,res_data
    else:
        res_data = []
        return last_date, res_data

if __name__ == '__main__':
    db = pymysql.connect(host='localhost', user='ystian', passwd='123456', db='ghtorrent_restore')
    cursor = db.cursor()
    project_id = 1088
    sql_0 = "select * from projects where id = " + str(project_id)
    last_date, res = Get('project_id', sql_0, cursor, 6, date(2011,1,1))
    print(res)
    print(last_date)