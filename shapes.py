import numpy as np
from matplotlib import pyplot as plt

#lets say size is 100 for now

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
            self.y, self.x = self.size / 4, self.size / 4
            self.maxY, self.maxX = self.y + self.size/2, self.x + self.size/2
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
            self.y, self.x = self.size / 4, self.size / 4
            self.maxY, self.maxX = self.y + self.size / 2, self.x + self.size / 2
            for x in range(self.size):
                for y in range(self.size):
                    if x >= self.x and x <= self.maxX:
                        if y >= x+1 and y <= self.maxX:
                            self.arr[x][y] = 'pos'
                        else:
                            self.arr[x][y] = 'neg'
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
