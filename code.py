# regole del percettrone:
#  moltiplico input per pesi
# sommo tutti i pesi
# computo l'output attraverso la funz di attivazione
#%%
from random import *
import numpy as np
def act(x) -> int:
    return 1 if x>=0 else -1

class Perceptron:
    
    def __init__(self,input): 
    #prende input e inizializza in base alla sua dimensione pesi randomici
        self.input = input
        self.weights = np.random.rand(len(input),1)
    
    def guess(self) -> int:
        sum=np.sum(self.input*self.weights)
        return act(sum)
x=np.array([1,2,2,5])
p=Perceptron(x)
print(p.guess())

