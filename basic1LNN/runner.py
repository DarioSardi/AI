#%%
import numpy as np 
import neuralNet as n
import matplotlib.pyplot as plt

X = np.array([[0,0,1,0,1,0,1],[0,1,1,0,0,0,0],[1,0,1,0,0,1,0],[1,1,1,1,1,1,1]])
y = np.array([[0],[1],[1],[0]])
nn = n.NeuralNet(7,1,4)
nn.trainLoop(X,y,2000)
show=False
if(show):
    plt.plot(nn.plotPointsX[10:],nn.plotPointsY1[10:],c='red')
    plt.plot(nn.plotPointsX[10:],nn.plotPointsY2[10:],c='blue')
    for ind,o in enumerate(nn.output):
        print("the answer "+str(ind)+" is "+str(o[0]))
    plt.show()

nn.answer([0,1,1,0,0,0,0])