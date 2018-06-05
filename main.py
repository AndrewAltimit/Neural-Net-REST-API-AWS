import sys, os
import time
# numpy and opencv
import numpy as np
import cv2
# sklearn
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

# HYPERPARAMETERS 
RES_WIDTH = 32
RES_HEIGHT = 32

LEARNING_RATE = 3e-4
EPOCHS = 100
BATCH_SIZE = 32
LOSS_FUNCTION = "CROSSENTROPY"

		
### FULLY CONNECTED NEURAL NETWORK ###
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
		
		# Layer Sizes
        N0 = 3072
        N1 = 25
        N2 = 1000
        Nout = 5
		
        # Create three fully connected layers, two of which are hidden and the third is
        # the output layer.  In each case, the first argument o Linear is the number
        # of input values from the previous layer, and the second argument is the number
        # of nodes in this layer.  The call to the Linear initializer creates a PyTorch
        # functional that in turn adds a weight matrix and a bias vector to the list of
        # (learnable) parameters stored with each Net object.  These weight matrices
        # and bias vectors are implicitly initialized using a normal distribution
        # with mean 0 and variance 1
        self.fc1 = nn.Linear( N0, N1, bias=True)
        self.fc2 = nn.Linear( N1, N2, bias=True)
        self.fc3 = nn.Linear( N2, Nout, bias=True)

    def forward(self, x):
        #  The forward method takes an input Variable and creates a chain of Variables
        #  from the layers of the network defined in the initializer. The F.relu is
        #  a functional implementing the Rectified Linear activation function.
        #  Notice that the output layer does not include the activation function.
        #  As we will see, that is combined into the criterion for the loss function.
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
		

### CONVOLUTIONAL NEURAL NETWORK ###
# http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
		
# Given a directory name and an extension of files to search for,
# the function will return a sorted list of files in the folder.
def get_image_paths_from_folder(dir_name, extension):
	# Store current working directory, then change to desired directory
	cwd = os.getcwd()
	os.chdir(dir_name)
	
	# Get the image paths in the folder with the requested extension
	img_list = os.listdir('./')
	img_list = [dir_name + "/" + name for name in img_list if extension.lower() in name.lower() ] 
	img_list.sort()
	
	# Restore the working directory
	os.chdir(cwd)
	
	return img_list
		
		
# Given an image path, return the feature vector for this image for the fully connected neural network
def process_image_FCNN(image_path):
	# Read Image
	image = cv2.imread(image_path)
	
	# Resize the Image
	resized_image = cv2.resize(image,(RES_WIDTH, RES_HEIGHT))
	
	# FCNN takes in the flattened data
	return resized_image.flatten().astype(np.float64)
	
	
# Given an image path, return the feature vector for this image for the convolutional neural network
def process_image_CNN(image_path):
	# Read Image
	image = cv2.imread(image_path)
	
	# Resize the Image
	resized_image = cv2.resize(image,(RES_WIDTH, RES_HEIGHT))
	
	# CNN takes in the data as a tensor: (samples, color channels, height, width)
	# so the color channels need to be brought to the front
	M, N, C = resized_image.shape
	return resized_image.reshape(C, M, N).astype(np.float64)
	
	
# Given a directory, return a list of sorted subdirectories
def get_subdirectories(directory):
	folders = []
	for i,j,y in os.walk(directory):
		if i == directory:
			continue
		folders.append(i)
	return sorted(folders)
	
def convert_to_categories(Y):
    _, categories = torch.max(Y.data, 1)
    categories = torch.Tensor.long(categories)
    return Variable(categories)
	
# Encode a label into an array of zeros with the label index being 1
def label_encoding(label, num_categories):
	data = [0] * num_categories
	data[label] = 1
	return data
	
	
# Given a dataset in dictionary format, reformat it to two lists, the classification list and the corresponding feature list
def format_dataset(dataset, msg, process_function):
	# Build Dataset
	X = []
	y = []
	num_labels = len(dataset.keys())
	for label in dataset:
		image_folder = dataset[label]
		for j in range(len(image_folder)):
			#print("{}... Label {} Image {}".format(msg, label, j))
			data = process_function(image_folder[j])
			X.append(data)
			y.append(label_encoding(label, num_labels))
			#print(label_encoding(label, num_labels))
	return X, y

	
# Given a training dataset, build the model and return the trained network and the criterion for use on the test dataset
def train(training_dataset, NN_class, process_function):
	# Format Training Dataset
	X, y = format_dataset(training_dataset, "TRAINING", process_function)
	sample_count = len(X)
			
	# Convert the feature vectors and labels to tensors
	X_train = Variable(torch.Tensor(X))
	Y_train = Variable(torch.Tensor(y))
	Y_trainc = convert_to_categories(Y_train)
	
	# Initialize FCNN / CNN
	net = NN_class()
	optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE)
	if LOSS_FUNCTION == "MSELoss":
		criterion = nn.MSELoss()
	elif LOSS_FUNCTION == "CROSSENTROPY":
		criterion = nn.CrossEntropyLoss()
	
	for ep in range(EPOCHS):
		#  Create a random permutation of the indices of the row vectors.
		indices = torch.randperm(sample_count)
		#  Run through each mini-batch
		for b in range(BATCH_SIZE):
			#  Use slicing (of the pytorch Variable) to extract the
			#  indices and then the data instances for the next mini-batch
			batch_indices = indices[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
			batch_X = X_train[batch_indices]
			batch_Y = Y_trainc[batch_indices]
			
			#  Run the network on each data instance in the minibatch
			#  and then compute the object function value
			pred_Y = net(batch_X)
			loss = criterion(pred_Y, batch_Y)
			
            #  Back-propagate the gradient through the network using the
            #  implicitly defined backward function, but zero out the
            #  gradient first.
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
            
		#  Print validation loss every 10 epochs
		if ep != 0 and ep%10==0:
			pred_Y = net(X_train)
			train_loss = criterion(pred_Y, Y_trainc)
			#print("Epoch %d loss: %.5f" %(ep, train_loss.data[0]))

	#  Compute and print the final training and test loss
	#  function values
	pred_Y_train = net(X_train)
	loss = criterion(pred_Y_train, Y_trainc)
	print('Final training loss is %.5f' %loss.data[0])
	print('Training success rate:', success_rate(pred_Y_train, Y_train))

	return net, criterion
	
	
# Given a testing dataset and a classifier, classify each sample and determine the overall error rate
def test(model, criterion, test_dataset, classifications, process_function, classes):
	X, y = format_dataset(test_dataset, "TESTING", process_function)
	
	# Convert the feature vectors and labels to tensors
	X_test = Variable(torch.Tensor(X))
	Y_test = Variable(torch.Tensor(y))
	Y_testc = convert_to_categories(Y_test)
	
	pred_Y_test = model(X_test)
	test_loss = criterion(pred_Y_test, Y_testc)
	print("Final test loss: %.5f" %test_loss.data[0])
	print('Test success rate:', success_rate(pred_Y_test, Y_test))
	
	# http://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
	# Keep track of correct guesses in a confusion matrix
	confusion = torch.zeros(len(classes), len(classes))
	
	# Go through a bunch of examples and record which are correctly guessed
	scores = [0] * len(classes)
	totals = [0] * len(classes)
	for i in range(len(pred_Y_test)):
		prediction = list(pred_Y_test[i].data)
		correct_answer = list(Y_test[i].data)
		guess_i = prediction.index(max(prediction))
		real_cat = correct_answer.index(max(correct_answer))
		
		confusion[real_cat][guess_i] += 1
		totals[real_cat] += 1
		if real_cat == guess_i:
			scores[real_cat] += 1
		
	print("\nCategory Order:")
	categories = [x.split("\\")[-1] for x in classes]
	print(", ".join(categories))
	print("\nConfusion Matrix:")
	print(confusion)
	print("Accuracy:")
	for i in range(len(categories)):
		print("Category {:<15} Accuracy: {:.1f}%".format(categories[i], 100 * (scores[i] / totals[i])))
	
	
def success_rate(pred_Y, Y):
    '''
    Calculate and return the success rate from the predicted output Y and the
    expected output.  There are several issues to deal with.  First, the pred_Y
    is non-binary, so the classification decision requires finding which column
    index in each row of the prediction has the maximum value.  This is achieved
    by using the torch.max() method, which returns both the maximum value and the
    index of the maximum value; we want the latter.  We do this along the column,
    which we indicate with the parameter 1.  Second, the once we have a 1-d vector
    giving the index of the maximum for each of the predicted and target, we just
    need to compare and count to get the number that are different.  We could do
    using the Variable objects themselve, but it is easier syntactically to do this
    using the .data Tensors for obscure PyTorch reasons.
    '''
    _,pred_Y_index = torch.max(pred_Y, 1)
    _,Y_index = torch.max(Y,1)
    num_equal = torch.sum(pred_Y_index.data == Y_index.data)
    num_different = torch.sum(pred_Y_index.data != Y_index.data)
    rate = num_equal / float(num_equal + num_different)
    return rate
	
	
# Given a list of folders and an extension of interest, return a dictionary of the image paths
# keys  -> classification label (number representing each unique folder)
# value -> list of image paths
def folder_list_to_image_dictionary(folders, extension):
	image_dictionary = dict()
	for i in range(len(folders)):
		image_dictionary[i] = get_image_paths_from_folder(folders[i], extension)
		
	return image_dictionary
	
	
if __name__ == "__main__":

	# Check if all proper input arguments exist
	if len(sys.argv) != 4:
		print("Improper number of input arguments")
		print("USAGE: prob4.py <train_dir> <test_dir> <mode (FCNN or CNN)>")
		sys.exit()
		
	start_time = time.time()
	mode = sys.argv[3].upper()
		
	# Get Subdirectories
	train_folders = get_subdirectories(sys.argv[1])
	test_folders = get_subdirectories(sys.argv[2])

	# Get dictionary of images: 
	training_dataset = folder_list_to_image_dictionary(train_folders, ".JPEG")
	test_dataset = folder_list_to_image_dictionary(test_folders, ".JPEG")
	
	print("{:=^100}".format("HYPERPARAMETERS"))
	if mode == "FCNN":
		print("Network Type: Fully Connected Neural Network")
	elif mode == "CNN":
		print("Network Type: Convolutional Neural Network")
	print("Image Resolution:", (RES_WIDTH, RES_HEIGHT))
	print("Learning Rate:", LEARNING_RATE)
	print("Epochs:", EPOCHS)
	print("Batch Size:", BATCH_SIZE)
	print("Loss Function:", LOSS_FUNCTION)
	print("=" * 100)

	if mode == "FCNN":
		### FCNN ###
		# Build the model with the training data
		model, criterion = train(training_dataset, Net, process_image_FCNN)
		
		# Evaluate the model with the testing data
		test(model, criterion, test_dataset, train_folders, process_image_FCNN, test_folders)
		
	elif mode == "CNN":
		### CNN ###
		# Build the model with the training data
		model, criterion = train(training_dataset, ConvNet, process_image_CNN)
		
		# Evaluate the model with the testing data
		test(model, criterion, test_dataset, train_folders, process_image_CNN, test_folders)
	
	end_time = time.time()
	print("Total Runtime: {:.2f} Seconds".format(end_time - start_time))
	

	
