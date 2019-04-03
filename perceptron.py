def act(x):
    if (x > 0):
        return 1
    else:
        return -1

import numpy as np
class Perceptron:
    
    def __init__(self): 
    	#prende input e inizializza in base alla sua dimensione pesi randomici
        self.weights = [np.random.uniform(-1,1),np.random.uniform(-1,1)]
        self.lr = 0.01
    def guess(self,inputs):
        sum2=0
        for i in range(0,len(self.weights)):
            sum2 += inputs[i] * self.weights[i];
        return act(sum2)
    
#w=w+err*input*learnRate
    def train(self,inputs,target):
        guessed = self.guess(inputs)
        error = target-guessed
        #print("p:",inputs,"g:",target,guessed)
        for i in range(0,len(self.weights)):
            self.weights[i] += error*inputs[i]*self.lr
#        print("")
