import numpy as np
import math
# load data from dataset and get a available information
def loadData():
    dataList = []
    with open('data.data','r') as file:
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

# Kmeans core scripts
def Kmeans(centers,data):
    row = data.shape[0]
    k = centers.shape[0]
    result = np.mat(np.zeros((row,1)))
    t = 0
    while 1:
        t += 1
        print(t)
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
    return result


# main function which is the start of the script
def main():
    # k classes need to be classify
    k = 3
    # get data matrix
    data = np.mat(loadData())
    # get random initial central point
    centers = getRandomPoint(k,data)
    result = Kmeans(centers,data)
    for j in range(k):
        print('the %d class is :' % j)
        item_class = data[np.nonzero(result[:, 0] == j)[0]]
        print(item_class.shape[0])

main()