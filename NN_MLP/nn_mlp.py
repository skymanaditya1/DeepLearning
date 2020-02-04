import numpy as np

features = np.array([[0,1,0], [0,0,1], [1,0,0], [1,1,0], [1,1,1]])
labels = np.array([[1,0,0,1,1]])
labels = labels.reshape(5,1)

# sigmoid function 
def sigmoid(z):
	return 1 / (1 + np.exp(-z))

# sigmoid derivative functioon 
def sigmoid_der(z):
	return sigmoid(z) * (1-sigmoid(z))

# initialization
weights = np.random.rand(features.shape[1], 1)
bias = np.random.rand(1)
alpha = 0.05

for i in range(20000):
	# forward propagation
	z = np.dot(features, weights) + bias
	a = sigmoid(z)

	# Squared loss 
	error = (a-labels)
	print("Epoch : " + str(i) + " Error : " + str(error.sum()))

	dz = error * sigmoid_der(z)
	dw = np.dot(features.transpose(), dz)
	db = dz

	# params updated rule 
	weights -= alpha/len(features)*dw
	bias -= alpha/len(features)*np.sum(db)

# inference
test = np.array([1,0,0])
predicted = sigmoid(np.dot(test, weights) + bias)
print("Predicted : " + str(predicted))

test = np.array([0,1,0])
predicted = sigmoid(np.dot(test, weights) + bias)
print("Predicted : " + str(predicted))