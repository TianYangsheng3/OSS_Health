import Question, Trend, Entropy
from Month_data import Month_one_duration, Month_all_duration, process_data


if __name__ == '__main__':
    labels = ['forks','committer','commits','commit_comment','req_opened','req_closed','req_merged','other','issue','issue_comment','watchers']
    rootpath = 'data\\'
    start, end = 1, 50
    components_num = 10
    have_std = False  

    ### 得到数据
    projects_valid, data = Month_all_duration(rootpath, start, end, have_std = have_std)
    data_proc, scalers = process_data(data, start, end)
    ### 得到每个月的pca权重
    pca = Trend.get_pca(data_proc, scalers, components_num)
    ### 得到每个月的信息熵方法的权重
    weights = Entropy.get_weights(data_proc, scalers, start, end, have_std = have_std)    


    #### 回答四个关键问题
    interval = [0.05, 0.15, 1]
    Question.Question_1(data_proc, projects_valid, scalers, weights, pca, start, end, have_std=have_std)
    Question.Question_2(data_proc, projects_valid, scalers, pca, start, end)
    Question.Question_3(data_proc, projects_valid, scalers, pca, start, end, have_std, interval)
    Question.Question_4(pca, start, end, have_std)