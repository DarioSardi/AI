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

########################  INIT THE SETS  ######################################
splitClouds=random.randint(0,120265)
TrainClouds,TestClouds = clouds[:splitClouds,:],clouds[splitClouds:,:]
splitBirds=random.randint(0,120265)
TrainBirds,TestBirds = birds[:splitBirds,:],clouds[splitBirds:,:]
splitEiffel=random.randint(0,120265)
TrainEiffel,TestEiffel = eiffel[:splitEiffel,:],clouds[splitEiffel:,:]
###############################################################################

def plotSome(a,columns,rows):
    fig=plt.figure(figsize=(28, 28))
    for i in range(1, columns*rows +1):
        img1 = 255-a[random.randint(0,120265)]
        img2 = np.split(img1,28)
        ax1 = fig.add_subplot(rows, columns, i,clip_on=False)
        ax1.axis('off')
        plt.imshow(img2,cmap='gray')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


#########################  NEURAL NET  #######################################
oracle = nn.NeuralNet(784,1,100)

birdAns = np.ones((splitBirds,1), dtype=np.int)*2
cloudAns = np.ones((splitClouds,1), dtype=np.int)*1
eiffelAns = np.ones((splitEiffel,1), dtype=np.int)*3

for k in range(10):
    print("begin training number ", k)
    oracle.trainLoop(TrainBirds,birdAns,10)
    print("     -training birds done")
    oracle.trainLoop(TrainEiffel,eiffelAns,10)
    print("     -training Eiffel done")
    oracle.trainLoop(TrainClouds,cloudAns,10)
    print("     -training clouds done")

validAnswers = {1: "cloud",2:"bird",3:"eiffel"}
for i in range(20):
    x=int(oracle.answer(TestBirds[i]))
    if x!=2 : print("wrong!")