# regole del percettrone:
#  moltiplico input per pesi
# sommo tutti i pesi
# computo l'output attraverso la funz di attivazione
# %%
import matplotlib.pyplot as plt
from random import *
import numpy as np
import perceptron
perc = perceptron.Perceptron(3)

RANGE = 1


def function(x):
    m = -1/3
    c = 0.5
    return m*x+c


def genFunction(x, y):
    if y > function(x):
        return 1
    else:
        return -1


class point:
    def __init__(self, x, y, b):
        self.pos = [x, y, b]
        self.group = genFunction(x, y)


def printPoints(pointsA):
    for p in pointsA:
        if (p.group == 1):
            plt.scatter(p.pos[0], p.pos[1], marker='s', c='black', s=100)
        else:
            plt.scatter(p.pos[0], p.pos[1], c='black', s=100)


def trainToPop(populationG, pa, number):
    for i in range(number):
        for dot in populationG:
            pa.train(dot.pos, dot.group)

        # print(pa.weights)
# POPULATE
population = [point(np.random.uniform(-RANGE, RANGE),
                    np.random.uniform(-RANGE, RANGE), 1) for i in range(100)]
# TRAINING


def printGuess(populationG, perc_):
    printPoints(populationG)
    for p in populationG:
        ans = perc_.guess(p.pos)
        if(ans == p.group):
            plt.scatter(p.pos[0], p.pos[1], c='g', s=20)
        else:
            plt.scatter(p.pos[0], p.pos[1], c='r', s=20)
    x = np.linspace(-RANGE, RANGE, 100)
    plt.plot(x, -(perc_.weights[0]/perc_.weights[1])
             * x-(perc_.weights[2]/perc_.weights[1]))
    plt.show()


printGuess(population, perc)
trainToPop(population, perc, 5)
x = np.linspace(-RANGE, RANGE, 100)
plt.plot(x, function(x), c='red')
printGuess(population, perc)
