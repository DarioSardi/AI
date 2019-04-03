# regole del percettrone:
#  moltiplico input per pesi
# sommo tutti i pesi
# computo l'output attraverso la funz di attivazione

from random import *
import numpy as np

import perceptron
pa=perceptron.Perceptron()

import points
import matplotlib.pyplot as plt
RANGE=10
def printPoints(pointsA):
    for p in pointsA:
        if (p.group==1):
            plt.scatter(p.pos[0],p.pos[1],marker='s',c='grey',s=100)
        else:
            plt.scatter(p.pos[0],p.pos[1],c='grey',s=100)

def trainToPop(populationG,pa,number):
    for i in range(number):
        for dot in populationG:
            pa.train(dot.pos,dot.group)
        print(pa.weights)
#POPULATE
population = [points.point(np.random.uniform(-RANGE,RANGE),np.random.uniform(-RANGE,RANGE)) for i in range(50)] 
# TRAINING
def printGuess(populationG,perc):
    printPoints(populationG)    
    for p in populationG:
        ans=perc.guess(p.pos)
        if(ans==p.group): plt.scatter(p.pos[0],p.pos[1],c='g',s=20)
        else: plt.scatter(p.pos[0],p.pos[1], c='r',s=20)
    x = np.linspace(-RANGE,RANGE, 100)
    plt.plot(x,-(pa.weights[0]/pa.weights[1])*x)
    plt.show()
printGuess(population,pa)
trainToPop(population,pa,5)
printGuess(population,pa)
