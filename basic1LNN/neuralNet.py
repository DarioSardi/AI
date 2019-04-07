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
    return 1.0/(1+ np.exp(-x))

def sigmoidD(x):
		return x * (1.0 - x)

class NeuralNet:

	def __init__(self, input_size,hidden_size,out_size):
		self.input = []
		self.iS =int(input_size)
		self.oS =int(out_size)
		# input weights matrice inputSize x 4 
		self.weightsI = np.random.random(( hidden_size,input_size))	# better init?
		# output weights matrice 4 x 1
		self.weightsO = np.random.random((out_size,hidden_size))		# better init?
		# inizializzo array output
		#self.bias_h = np.ones((hidden_size, 1));
		#self.bias_o = np.ones((hidden_size, 1));
		self.lr = 0.1
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
		self.layerI = sigmoid(np.dot(self.weightsI,self.input))
		# output results
		self.output = sigmoid(np.dot(self.weightsO,self.layerI))

	def backprop(self):
		# calcolo i valori delle loss F.
		b=self.lr*(self.y - self.output) * sigmoidD(self.output)
		a=np.dot( self.weightsO.T,b) * sigmoidD(self.layerI)
		updateI = np.dot(a,self.input.T)
		updateO = np.dot((self.lr*(self.y - self.output)* sigmoidD(self.output)),self.layerI.T)
		
		self.errorI=updateI
		self.errorO=updateO
	
		self.weightsI += updateI
		self.weightsO += updateO
	
	def train(self):
		hidd = np.dot(self.input,self.weightsI)
		#hidd += self.bias_h
		hidd = sigmoid(hidd)
		self.output =np.dot(hidd,self.weightsO)#+self.bias_o
		self.output = sigmoid(self.output)
		#output error
		o_error = self.y - self.output
		gradients = np.dot(o_error*sigmoidD(self.output),self.lr)
		weight_ho_deltas=np.dot(hidd.T,gradients)
		#adjust out weights
		self.weightsO += weight_ho_deltas
		#self.bias_o += gradients
		#hidden layer error
		hidden_errors =np.dot(self.weightsO.T,o_error)
		hidden_gradient = np.dot(np.dot(sigmoidD(hidd),hidden_errors),self.lr)
		weight_ih_deltas = np.dot(hidden_gradient,self.input.T)
		self.weightsI += weight_ih_deltas + hidden_gradient
		


	def trainOnce(self,input_,output_):
		self.y = output_		
		self.input = input_
		self.ff()
		self.backprop()

	def trainLoop(self,input_,output_,cycle):
		#check dimens.
		if (input_.shape[1]==self.iS and output_.shape[1]==self.oS):
			self.y = output_		
			self.input = input_
			for i in range(cycle):
				#train
				self.ff()
				self.backprop()
				#data vis. array
				self.plotPointsX.append(i)
				self.plotPointsY1.append(np.sum(self.errorI))
				self.plotPointsY2.append(np.sum(self.errorO))
		else: 
			print("WRONG IO SIZES!")
			print("input:",input_.shape[1]==self.iS)
			print("output",output_.shape[1]==self.oS)
	
	def answer(self,input_):
		self.input=input_
		self.ff()
		print(self.output)
		return self.output
	
	def plotError(self):
		plt.plot(self.plotPointsX[10:],self.plotPointsY1[10:],c='red')
		plt.plot(self.plotPointsX[10:],self.plotPointsY2[10:],c='blue')
		plt.show()