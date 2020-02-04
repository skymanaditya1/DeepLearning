import numpy as np

def sigmoid(z):
	return 1 / (1+np.exp(-z))

def sigmoid_prime(z):
	return sigmoid(z)*(1-sigmoid(z))

class NeuralNetwork(object):
	def __init__(self, X, Y):
		self.alpha = 0.05
		self.inputs = X
		self.Y = Y
		self.output = np.zeros(self.Y.shape)
		self.hlayer1_size = 10
		self.weights1 = np.random.rand(self.inputs.shape[1], self.hlayer1_size)
		self.weights2 = np.random.rand(self.hlayer1_size, self.output.shape[1])
		self.bias1 = np.random.rand(1)
		self.bias2 = np.random.rand(1)

	def forward_propagation(self):
		self.layer1 = sigmoid(np.dot(self.inputs, self.weights1) + self.bias1)
		self.output = sigmoid(np.dot(self.layer1, self.weights2) + self.bias2)
		return self.output

	def backward_propagation(self):
		self.error = (self.output - self.Y)
		self.odelta = self.error * sigmoid_prime(self.output)
		self.dweights2 = np.dot(self.layer1.transpose(), self.odelta)
		self.dbias2 = self.odelta
		self.weights2 -= self.alpha/self.inputs.shape[0]*self.dweights2
		self.bias2 -= self.alpha/self.inputs.shape[0]*np.sum(self.dbias2)

		self.l1error = np.dot(self.odelta, self.weights2.transpose())
		self.l1delta = self.l1error * sigmoid_prime(self.layer1)
		self.dweights1 = np.dot(self.inputs.transpose(), self.l1delta)
		self.dbias1 = self.l1delta
		self.weights1 -= self.alpha/self.inputs.shape[0]*self.dweights1 
		self.bias1 -= self.alpha/self.inputs.shape[0]*np.sum(self.dbias1)

	def saveWeights(self):
		np.savetxt("params\\w1.txt", self.weights1, fmt="%s")
		np.savetxt("params\\w2.txt", self.weights2, fmt="%s")
		np.savetxt("params\\b1.txt", self.bias1, fmt="%s")
		np.savetxt("params\\b2.txt", self.bias2, fmt="%s")

def predict(input, weights1, bias1, weights2, bias2):
	layer1 = sigmoid(np.dot(input, weights1) + bias1)
	output = sigmoid(np.dot(layer1, weights2) + bias2)
	return output

def load_params(filename):
	param = list()
	with open(filename, 'r') as f:
		for line in f.read().split('\n'):
			row = [float(x) for x in line.split(' ')]
			param.append(row)
	return np.array(param)

def load_param(filename):
	with open(filename, 'r') as f:
		param = [float(x) for x in f.read().split('\n')]
	return np.array(param)

if __name__ == "__main__":
	'''epochs = 20000
	features = np.array([[0,1,0], [0,0,1], [1,0,0], [1,1,0], [1,1,1]])
	labels = np.array([[1,0,0,1,1]])
	labels = labels.reshape(5,1)

	NN = NeuralNetwork(features, labels)

	for i in range(epochs):
		predicted = NN.forward_propagation()
		error = (predicted - labels)
		print("Epoch : " + str(i) + ", Error : " + str(error.sum()))
		NN.backward_propagation()

	NN.saveWeights()'''

	weights1 = load_params('params\\w1.txt')
	weights2 = load_param('params\\w2.txt')
	bias1 = load_param('params\\b1.txt')
	bias2 = load_param('params\\b2.txt')

	test = np.array([[1,0,0]])
	predicted = predict(test, weights1, bias1, weights2, bias2)
	print(predicted)

	test = np.array([[0,1,0]])
	predicted = predict(test, weights1, bias1, weights2, bias2)
	print(predicted)