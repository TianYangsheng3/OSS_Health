import numpy as np 
from datetime import datetime, date, timedelta
from sklearn.preprocessing import normalize
import random
from sklearn.decomposition import PCA, SparsePCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

'''
采用主成分分析得到项目的健康性评分
'''


def pca_1(data, num_components):    # 普通的主成分分析,这里的num_components暂时用不到
    # n_components：这个参数可以帮我们指定希望PCA降维后的特征维度数目。
    # 最常用的做法是直接指定降维到的维度数目，此时n_components是一个大于等于1的整数。
    # 当然，我们也可以指定主成分的方差和所占的最小比例阈值，让PCA类自己去根据样本特征方差来决定降维到的维度数，此时n_components是一个（0，1]之间的数。
    # 当然，我们还可以将参数设置为"mle", 此时PCA类会用MLE算法根据特征的方差分布情况自己去选择一定数量的主成分特征来降维。
    # 我们也可以用默认值，即不输入n_components，此时n_components=min(样本数，特征数)。
    pca = PCA(n_components = 'mle') 
    # pca = PCA() 
    pca.fit(data)
    # print(pca.n_components_)
    # print("explained_variance_ratio_: ", [round(v,4) for v in pca.explained_variance_ratio_])
    # # print(pca.explained_variance_)
    # # print(pca.components_)
    return pca

def compute_newx(pca, X, is_normalize = False):
    # if is_normalize == True:
    #     X = normalize(X = X, axis=0)
    X = np.array(X)
    new_X = pca.transform(X)
    print("new_X: ", new_X)
    weights = list(pca.explained_variance_ratio_)
    score = []
    for project in new_X:
        tmp = 0
        for i in range(len(weights)):
            tmp += weights[i]*project[i]
        score.append(tmp)
    return score    

def compute(pca, X, is_normalize = False):
    weights = pca.explained_variance_ratio_
    project = np.array(X)
    components = pca.components_
    # score = np.dot(np.dot(weights, components), project)
    indi_weights = np.dot(weights, components)
    tmp = indi_weights.sum()
    for i in  range(len(indi_weights)):
        indi_weights[i] = indi_weights[i]/tmp
    # score = np.dot(indi_weights, project)
    scores = []
    for i in range(len(project)):
        score = np.dot(indi_weights, project[i])
        scores.append(score)

    return scores   

def LinearReg(X, y):
    reg = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    reg.fit(X=X_train, y=y_train)
    print("coef: ", reg.coef_)
    y_pred = reg.predict(X_test)
    tmp = np.array(range(len(X_test)))
    plt.plot(tmp, y_test, label = 'y')
    plt.plot(tmp, y_pred, label = 'y_pred')
    plt.legend()
    plt.show()


def LinearFit(X, y):
    reg = LinearRegression()
    X_train = np.array(X).reshape(-1, 1)
    y_train = np.array(y).reshape(-1, 1)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    reg.fit(X=X_train, y=y_train)
    coef = reg.coef_
    intercept = reg.intercept_
    # print("coef: ", reg.coef_)
    y_pred = reg.predict(np.array(X_train))
    # print(y_pred.shape)
    return y_pred
    # plt.plot(X, y_pred, label = 'fit')
    # plt.legend()
    # plt.show()

def Coef(X, y):
    reg = LinearRegression()
    X_train = np.array(X).reshape(-1, 1)
    y_train = np.array(y).reshape(-1, 1)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    reg.fit(X=X_train, y=y_train)
    coef = reg.coef_
    intercept = reg.intercept_
    # print("coef: ", reg.coef_)
    # y_pred = reg.predict(np.array(X_train))
    # print(coef.shape)
    return coef

def compute_selects(data, selects, projects_valid, pca):    # 普通的主成分分析,这里的num_components暂时用不到
    X = data[:]
    X = normalize(X = X, axis=0)
    X = np.array(X)
    new_X = pca.transform(X)
    weights = list(pca.explained_variance_ratio_)
    for i in range(len(selects)):
        print("*************************** scores of cluster " + str(i) + " ***************************")
        for v in selects[i]:
            score = 0
            for k in range(len(weights)):
                score += weights[k]*new_X[v][k]
            print("project_id "+ str(projects_valid[v]) + " scores is " + str(round(score,4)))
