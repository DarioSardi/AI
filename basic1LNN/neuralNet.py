"""
One layer Neural Network
=====
Provides a simple 1 hidden layer neural network.
Initialize it with input data size,output data size and number of nodes in the hidden layer.
"""
import numpy as np

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoidD(x):
		return x * (1.0 - x)

class NeuralNet:

	def __init__(self, input_size, out_size,hidden_size):
		self.input = []
		self.iS =int(input_size)
		self.oS =int(out_size)
		# input weights matrice inputSize x 4 
		self.weightsI = np.random.rand(input_size,hidden_size)	# better init?
		# output weights matrice 4 x 1
		self.weightsO = np.random.rand(hidden_size, 1)			# better init?
		# inizializzo array output
		self.output = np.zeros(out_size)
		self.plotPointsX = []
		self.plotPointsY1 = []
		self.plotPointsY2 = []

	def ff(self):
		# layer1 results
		self.layerI = sigmoid(np.dot(self.input, self.weightsI))
		# output results
		self.output = sigmoid(np.dot(self.layerI, self.weightsO))

	def backprop(self):
		# calcolo i valori delle loss F.
		updateI = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoidD(self.output), self.weightsO.T) * sigmoidD(self.layerI)))
		updateO = np.dot(self.layerI.T, (2*(self.y - self.output)* sigmoidD(self.output)))
		
		self.errorI=updateI
		self.errorO=updateO
	
		self.weightsI += updateI
		self.weightsO += updateO
	
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