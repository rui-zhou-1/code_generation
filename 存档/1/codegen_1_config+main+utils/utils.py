# Implement this file for a project that: 生成决策树分类代码


import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import time
import cPickle as pickle
import sys

def load_data(filename):
    
    # 打开文件
    fr = open(filename)
    # 将文件内容读入内存
    arrayOLines = fr.readlines()
    # 得到文件行数
    numberOfLines = len(arrayOLines)
    # 返回的特征矩阵
    returnMat = np.zeros((numberOfLines, 3))
    # 将每一行以\t为分隔符，分成三部分，并存入数组
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    """单层决策树分类函数
    input:  dataMatrix(list):数据集
            dimen(int):第dimen维特征
            threshVal(float):阈值
            threshIneq(