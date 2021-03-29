import numpy as np
from matplotlib import pyplot as plt

#lets say size is 100 for now

def signForPoint(p1, p2, p3):
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

def pointInTri(pt, v1, v2, v3):
    d1 = signForPoint(pt, v1, v2)
    d2 = signForPoint(pt, v2, v3)
    d3 = signForPoint(pt, v3, v1)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


class Mtx:
    def __init__(self, size):
        self.size = size
        self.arr = np.empty((self.size, self.size), dtype=object)
        self.x = 0
        self.y = 0
        self.maxX = size-1
        self.maxY = size-1
        self.shape = ''

    def setShape(self, shape):
        self.shape = shape
        if shape == 'square':
            self.y, self.x = self.size / 7, self.size / 7
            self.maxY, self.maxX = self.y + self.size/1.4, self.x + self.size/1.4
            for x in range(self.size):
                for y in range(self.size):
                    if x >= self.x and x <= self.maxX:
                        if y >= self.y and y <= self.maxY:
                            self.arr[x][y] = 'pos'
                        else:
                            self.arr[x][y] = 'neg'
                    else:
                        self.arr[x][y] = 'neg'

        elif shape == 'triangle':
            self.p1 = [0, 0] #self.size/20, self.size/20
            self.p2 = [self.size/2, self.size - 1]
            self.p3 = [self.size - 1, 0]
            #self.y, self.x = self.size / 4, self.size / 4
            #self.maxY, self.maxX = self.y + self.size / 1.5, self.x + self.size / 1.5
            for x in range(self.size):
                for y in range(self.size):
                    if pointInTri([x, y], self.p1, self.p2, self.p3):
                        self.arr[x][y] = 'pos'
                    else:
                        self.arr[x][y] = 'neg'

        elif shape == 'circle':
            self.y, self.x = self.size / 2, self.size / 2
            self.radius = self.size / 3
            for x in range(self.size):
                for y in range(self.size):
                    if (x-self.x)**2 + (y-self.y)**2 <= self.radius**2:
                        self.arr[x][y] = 'pos'
                    else:
                        self.arr[x][y] = 'neg'
        elif shape == 'crazy':
            pass

    def showMat(self):
        plt.matshow(self.arr)
        plt.show()

    def returnData(self):
        data = np.empty((self.size, self.size), dtype=object)
        if self.shape != 'circle':
            for x in range(self.size):
                for y in range(self.size):
                    data[x][y] = [x, y, self.arr[x][y]]
        else:
            for x in range(self.size):
                for y in range(self.size):
                    data[x][y] = [x, y,  self.arr[x][y]] # (x-self.x)**2 + (y-self.y)**2,self.radius**2,
        #print(data.shape)
        #data = np.reshape(data, (self.size+self.size, 3, -1))
        #print(data.shape)
        #data.transpose(2, 0, 1).reshape(3, -1)
        return data





#a = Mtx(1000)
#a.setShape("circle.pdf")
#a.showMat()
