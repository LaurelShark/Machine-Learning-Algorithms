from random import uniform, seed
from itertools import product
from math import sqrt, pow, floor
import matplotlib.pyplot as plt

seed(9001)

def generateClasses(maxx1, maxx2): # TODO consider deleting, for it is never used
    return [ cls for cls in product(range(maxx1), range(maxx2)) ]
def generateInstances(l, maxx1, maxx2):
    sample = []
    for i in range(l):
        x1 = uniform(0, maxx1)
        x2 = uniform(0, maxx2)
        sample.append( (x1, x2) )
    return sample
def getDistance(a, b): # returns distance between two points
    x1, y1 = a
    x2, y2 = b
    return sqrt( pow(x2 - x1, 2) + pow(y2 - y1, 2) )
def getClass(x): # returns actual class
    return ( floor(x[0]), floor(x[1]) )
def getClassByClosestNeighbors(x, sample, k=1, withGraphics=True): # returns supposed class by closest neighbors
    closest = sorted(sample, key=lambda o: getDistance(x, o), reverse=False)[:k]
    if withGraphics:
        fig = plt.figure()
        ax1 = fig.add_subplot(221)
        ax1.scatter(*zip(*sample), marker=r'$\clubsuit$') # show all objects
        ax1.plot(x[0],x[1],'r*') # show the object x
        ax2 = fig.add_subplot(222)
        ax2.scatter(*zip(*closest), marker='+') # show only closest objects
        ax2.plot(x[0],x[1],'r*') # show the object x
        axes = plt.gca()
        axes.set_xlim([0,10]) # limit axes, hardcoded values
        axes.set_ylim([0,10])
        plt.show()
    classDict = {}	
    for item in closest:
        cls = getClass( item )
        classDict[cls] = classDict.get(cls, 0) + 1
    maxCls, maxCount = 0, 0
    for (cls, val) in classDict.items():
        if maxCount < val:
            maxCount = val
            maxCls = cls
    return maxCls
def getLoss(x, sample, k):
    actualCls = getClass(x)
    supposedCls = getClassByClosestNeighbors(x, sample, k, withGraphics=False)
    return getDistance(actualCls, supposedCls)
def getQuality(sample, k):
    sum = 0
    for elt in sample:
        sum += getLoss(elt, sample, k)
    return sum

instances = generateInstances(250, 10, 10)
print(getClassByClosestNeighbors((2.7, 4.3), instances, k=10))
qualityForK = []
for k in range(1,10):
    print("calculating quality function for k=" + str(k))
    qualityForK.append( (k, getQuality(instances, k)) )
print("(k,quality function)")
print(sorted(qualityForK, key=lambda a: a[1]))

