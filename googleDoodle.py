
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
splitClouds=int(120265*0.75)#random.randint(0,120265)
TrainClouds,TestClouds = np.split(clouds,[splitClouds])
splitBirds=int(133572*0.75)#random.randint(0,133572)
TrainBirds,TestBirds = np.split(birds,[splitBirds])
splitEiffel=int(134801*0.75)#random.randint(0,134801)
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
def plot(img):
       plt.imshow(img)#,cmap='gray')
       plt.show()

#plotSome(eiffel,20,20)


#########################  NEURAL NET  #######################################
import basic1LNN.neuralNet as nn
oracle = nn.NeuralNet(784,64,3)
train=False
if train:
        trainingSet= []
        print("generating training set")
        for i in range(splitBirds-1):
               x=np.reshape(TrainBirds[i]/255.0,(784,1))
               y=np.reshape(np.array([1,0,0]),(3,1))
               trainingSet.append([x,y])
        for i in range(splitClouds-1):
               x=np.reshape(TrainClouds[i]/255.0,(784,1))
               y=np.reshape(np.array([0,1,0]),(3,1))
               trainingSet.append([x,y])
        for i in range(splitEiffel-1):
               x=np.reshape(TrainEiffel[i]/255.0,(784,1))
               y=np.reshape(np.array([0,0,1]),(3,1))
               trainingSet.append([x,y])
        r = random.SystemRandom()
        r.shuffle(trainingSet)
        print("let the training begin")
        for el in trainingSet:
                oracle.trainOnce(el[0],el[1])
        oracle.export()
else:
        oracle.importPar()
wrong=0
test=False
if test:
        for i in range(TestBirds.shape[0]):
                ans=oracle.answer(np.reshape(TestBirds[i]/255.0,(784,1)))
                if max(ans[0],ans[1],ans[2])==ans[0]:
                        continue
                else:
                        #print(ans)
                        wrong+=1
        print(str(100*wrong/i)+" percento di errori su "+str(i)+" disegni")

#MYINPUT
img = np.array(Image.open("myInput.png").convert('L'))
img = 255-img
plt.imshow(img,cmap='gray')
plt.show()
#print(img)
ans=oracle.answer(np.reshape(img/255.0,(784,1)))
def whatIs(a):
        if (a[0]>a[1] and a[0]>a[2]):
                print("un uccello?!")
        elif (a[1]>a[0] and a[1]>a[2]):
                print("una nuvola?!")
        else:
                print("la torre eiffel?!")
whatIs(ans)