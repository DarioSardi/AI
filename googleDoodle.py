#%%
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import random
import basic1LNN.neuralNet  as nn

#########################  IMPORT SETS  #######################################
clouds=np.load("dataset/cloud.npy") #28x28 img
birds=np.load("dataset/bird.npy") #28x28 img
eiffel=np.load("dataset/eiffel.npy") #28x28 img
hotdog=np.load("dataset/hot-dog.npy") #28x28 img

########################  INIT THE SETS  ######################################
splitClouds=random.randint(0,120265)
TrainClouds,TestClouds = np.split(clouds,[splitClouds])
splitBirds=random.randint(0,133572)
TrainBirds,TestBirds = np.split(birds,[splitBirds])
splitEiffel=random.randint(0,134801)
TrainEiffel,TestEiffel = np.split(eiffel,[splitEiffel])
###############################################################################

def plotSome(a,columns,rows):
    fig=plt.figure(figsize=(28, 28))
    for i in range(1, columns*rows +1):
        img1 = 255-a[random.randint(0,len(a))]
        img2 = np.split(img1,28)
        ax1 = fig.add_subplot(rows, columns, i,clip_on=False)
        ax1.axis('off')
        plt.imshow(img2,cmap='gray')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

#plotSome(hotdog,20,20)


#########################  NEURAL NET  #######################################
import basic1LNN.neuralNet as nn
oracle = nn.NeuralNet(784,100,3)
for k in range(20):
    for i in range(100):
        x=np.reshape(TrainBirds[i],(784,1))
        y=np.reshape(np.array([1,0,0]),(3,1))
        oracle.trainOnce(x,y)
        x=np.reshape(TrainClouds[i],(784,1))
        y=np.reshape(np.array([0,1,0]),(3,1))
        oracle.trainOnce(x,y)
        x=np.reshape(TrainEiffel[i],(784,1))
        y=np.reshape(np.array([0,0,1]),(3,1))
        oracle.trainOnce(x,y)
oracle.answer(TestBirds[4])