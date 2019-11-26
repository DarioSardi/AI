"""
One layer Neural Network
=====
Provides a simple 1 hidden layer neural network.
__init__(input_size,output_size,hidden_layer_size) \n
input_size : number of parameters passed as inputs for training or guessing \n
output_size : number of output parameters expected \n
hidden_layer_size : number of hidden layer neurons \n
"""
#%%

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
	ans= 1.0/(1+ np.exp(-x))
	return ans

def sigmoidD(x):
	ans= x * (1.0 - x)
	return ans
class NeuralNet:

	def __init__(self, input_size,hidden_size,out_size):
		self.input = []
		self.iS =int(input_size)
		self.oS =int(out_size)
		# input weights matrice inputSize x 4 
		self.weightsI = np.random.random(( hidden_size,input_size))*2-1		# better init?
		# output weights matrice 4 x 1
		self.weightsO = np.random.random((out_size,hidden_size))*2-1		# better init?
		#inizializzo bias tra -1 e 1. random va da 0 a 1
		self.bias_h =np.random.random((hidden_size,1))*2-1
		self.bias_o = np.random.random((out_size,1))*2-1
		self.lr = 0.1
		# inizializzo array output
		self.output = np.zeros(out_size)
		self.plotPointsX = []
		self.plotPointsY1 = []
		self.plotPointsY2 = []

	def setWeights(self,w1,w2):
		self.weightsI=w1
		self.weightsO=w2
	
	def getWeights(self):
		print(self.weightsI.shape[0],self.weightsI.shape[1])
		print(self.weightsO.shape[0],self.weightsO.shape[1])

	def ff(self):
		# layer1 results
		#print(self.weightsI.shape,self.input.shape)
		self.hidden = sigmoid(np.dot(self.weightsI,self.input)+self.bias_h)
		# output results
		self.output = sigmoid(np.dot(self.weightsO,self.hidden)+self.bias_o)


	def backprop(self):
		out_err = self.y-self.output
		gradiente_o = (sigmoidD(self.output)*out_err)*self.lr
		delta_ho = np.dot(gradiente_o,self.hidden.T)
		self.weightsO+=delta_ho
		self.bias_o+=gradiente_o
		gradiente_h = sigmoidD(self.hidden)*np.dot(self.weightsO.T,out_err)*self.lr
		delta_ih = np.dot(gradiente_h , self.input.T)
		self.weightsI += delta_ih
		self.bias_h += gradiente_h

	def trainOnce(self,input_,output_):
		self.y = output_		
		self.input = input_
		self.ff()
		self.backprop()

	def answer(self,input_):
		self.input=input_
		self.ff()
		#print(np.round(self.output, 1))
		return np.round(self.output, 3)
	
	def plotError(self):
		plt.plot(self.plotPointsX[10:],self.plotPointsY1[10:],c='red')
		plt.plot(self.plotPointsX[10:],self.plotPointsY2[10:],c='blue')
		plt.show()
	
	def export(self):
		np.save("dataset/Win", self.weightsI)
		np.save("dataset/Wo", self.weightsO)
		np.save("dataset/biasH", self.bias_h)
		np.save("dataset/biasO", self.bias_o)
		print("EXPORTED")
	def importPar(self):
		self.weightsI=np.load("dataset/Win.npy")
		self.weightsO=np.load("dataset/Wo.npy")
		self.bias_h=np.load("dataset/biasH.npy")
		self.bias_o=np.load("dataset/biasO.npy")
	