
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import random
import basic1LNN.neuralNet  as nn
TRAIN=False
TEST=True
PLOT=False

if TRAIN or TEST:
#########################  IMPORT SETS  #######################################
	clouds=np.load("dataset/cloud.npy") #28x28 img
	birds=np.load("dataset/bird.npy") #28x28 img
	eiffel=np.load("dataset/eiffel.npy") #28x28 img

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
birdsE=[]
cloudsE=[]
eiffelE=[]
yE=[]
for i in range(1):
        yE.append(i)
        print(i)
        oracle = nn.NeuralNet(784,64,3)
        if TRAIN:
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

        if TEST:
                for i in range(TestBirds.shape[0]):
                        ans=oracle.answer(np.reshape(TestBirds[i]/255.0,(784,1)))
                        if max(ans[0],ans[1],ans[2])==ans[0]:
                                continue
                        else:
                                #print(ans)
                                wrong+=1
                birdsE.append(100*wrong/i)
                print(str(int(100*wrong/i))+" percento di errori su "+str(i)+" uccelli")
                wrong=0
                for i in range(TestClouds.shape[0]):
                        ans=oracle.answer(np.reshape(TestClouds[i]/255.0,(784,1)))
                        if max(ans[0],ans[1],ans[2])==ans[1]:
                                continue
                        else:
                                #print(ans)
                                wrong+=1
                cloudsE.append(100*wrong/i)
                print(str(int(100*wrong/i))+" percento di errori su "+str(i)+" nuvole")
                wrong=0
                for i in range(TestEiffel.shape[0]):
                        ans=oracle.answer(np.reshape(TestEiffel[i]/255.0,(784,1)))
                        if max(ans[0],ans[1],ans[2])==ans[2]:
                                continue
                        else:
                                #print(ans)
                                wrong+=1
                eiffelE.append(100*wrong/i)
                print(str(int(100*wrong/i))+" percento di errori su "+str(i)+" eiffel")
                wrong=0
if PLOT:
        plt.scatter(yE,birdsE,label='uccelli')
        plt.scatter(yE,cloudsE,label='nuvole')
        plt.scatter(yE,eiffelE,label='torre eiffel')
        plt.ylabel("% errore")
        plt.xlabel("test numero")
        plt.legend()
        plt.show()
#MYINPUT
img = np.array(Image.open("myInput.png").convert('L'))
plt.imshow(img,cmap='gray')
legend="1,0,0=uccello\n0,1,0=nuvola\n0,0,1=torre eiffel"
plt.text(33 ,3,legend,horizontalalignment='center',verticalalignment='center')
#print(img)
ans=oracle.answer(np.reshape(img/255.0,(784,1)))
def whatIs(a):
        global plt
        ansString="["+str(a[0][0])+","+str(a[1][0])+","+str(a[2][0])+"]"
        plt.text(15, 2.5, ansString, horizontalalignment='center')
        if (a[0]>a[1] and a[0]>a[2]):
                plt.text(15, 5, 'è un uccello?', horizontalalignment='center')
                #print("un uccello?!")
        elif (a[1]>a[0] and a[1]>a[2]):
                plt.text(15,5,'è una nuvola?',horizontalalignment='center')
                #print("una nuvola?!")
        else:
                plt.text(15, 5, 'è la torre eiffel?', horizontalalignment='center')
                #print("la torre eiffel?!")
whatIs(ans)
plt.show()
