import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# load data from dataset and get a available information
def loadData(file):
    dataList = []
    with open(file,'r') as file:
        data = file.readlines()
        for line in data:
            dom = line.split(',')[:-1]
            dataList.append(dom)
    return np.array(dataList).astype(float)

# get random points
def getRandomPoint(k,data):
    columns = data.shape[1]
    points = np.mat(np.zeros((k,columns)))
    for c in range(columns):
        min_cols = min(data[:,c].A)[0]
        max_cols = max(data[:,c].A)[0]
        range_cols = max_cols - min_cols
        points[:,c] = min_cols + range_cols*np.random.rand(k,1)
    return points

# caculate the Oura distance
def getDistance(a,b):
    return sum((a - b)**2)

def showResult(data,centers,result):
    k = centers.shape[0]
    count = data.shape[0]
    marks = ['ob','og','oc','om','oy','ok']
    for i in range(count):
        index = result[i,0]
        mark = marks[index]
        plt.plot(data[i,0],data[i,1],mark)
    for i in range(k):
        plt.plot(centers[i,0],centers[i,1], 'Dr', markersize = 10)
    plt.show()

# Kmeans core scripts
def Kmeans(centers,data):
    row = data.shape[0]
    k = centers.shape[0]
    result = np.mat(np.zeros((row,1)).astype(int))
    while 1:
        lceters = centers.copy()
        for i in range(row):
            item = data.A[i,:]
            classes = -1
            min_dis = math.inf
            for c in range(k):
                center = centers.A[c,:]
                dis = getDistance(item,center)
                if min_dis>dis:
                    min_dis = dis
                    classes = c
            result[i:]= classes
        # change the center points
        for j in range(k):
            item_class = data[np.nonzero(result[:,0] == j)[0]]
            centers[j,:] = np.mean(item_class,0)
        # if the center points won't change any further, we could know that we have get the optimize classes
        if (lceters == centers).all():
            break
    return result,centers


# main function which is the start of the script
def test(file,k):
    # get data matrix
    data = loadData(file)
    pca = PCA(n_components=2)
    data = np.mat(pca.fit_transform(data))
    # get random initial central point
    centers = getRandomPoint(k,data)
    result,centers = Kmeans(centers,data)
    showResult(data,centers,result)

# first param is data file path and second param is k classes need to be classify
test('iris.csv',3)
test('rank.csv',5)