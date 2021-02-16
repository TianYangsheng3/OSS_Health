import numpy as np 
from datetime import datetime, date, timedelta
from sklearn.preprocessing import normalize
import random
from sklearn.decomposition import PCA, SparsePCA
from Month_data import Month_all

def pca_1(data, num_components, selects, projects_valid):    # 普通的主成分分析,这里的num_components暂时用不到
    X = data[:]
    X = normalize(X = X, axis=0)
    X = np.array(X)
    # n_components：这个参数可以帮我们指定希望PCA降维后的特征维度数目。
    # 最常用的做法是直接指定降维到的维度数目，此时n_components是一个大于等于1的整数。
    # 当然，我们也可以指定主成分的方差和所占的最小比例阈值，让PCA类自己去根据样本特征方差来决定降维到的维度数，此时n_components是一个（0，1]之间的数。
    # 当然，我们还可以将参数设置为"mle", 此时PCA类会用MLE算法根据特征的方差分布情况自己去选择一定数量的主成分特征来降维。
    # 我们也可以用默认值，即不输入n_components，此时n_components=min(样本数，特征数)。
    # pca = PCA(n_components = 'mle') 
    pca = PCA() 
    pca.fit(X)
    new_X = pca.transform(X)
    weights = list(pca.explained_variance_ratio_)
    # print(pca.n_components_)
    print("explained_variance_ratio_: ", [round(v,4) for v in pca.explained_variance_ratio_])
    # print(pca.explained_variance_)
    # print(pca.components_)
    for i in range(len(selects)):
        print("*************************** scores of cluster " + str(i) + " ***************************")
        for v in selects[i]:
            score = 0
            for k in range(len(weights)):
                score += weights[k]*new_X[v][k]
            print("project_id "+ str(projects_valid[v]) + " scores is " + str(round(score,4)))

