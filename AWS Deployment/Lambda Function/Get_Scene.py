import boto3
import sys, os
import time
import base64

### Pillow ###
from PIL import Image

### Numpy ###
import numpy as np

### PyTorch ###
try:
	# Provides all PyTorch dependencies to the tmp folder to overcome the 250MB size limit.
    import unzip_requirements
except ImportError:
    pass
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F


### Global Variables ###
RES_WIDTH = 32
RES_HEIGHT = 32
# Alphabetical list of classes the model is based on
MODEL_CLASSES = ["grass", "ocean", "redcarpet", "road", "wheatfield"]

		
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
		
		
		
# Given an image path, return the feature vector for this image for the fully connected neural network
def process_image_FCNN(image_path):
	# Read Image
	image = Image.open(image_path)
	
	# Resize the Image and convert to numpy array
	resized_image = image.resize((RES_WIDTH, RES_HEIGHT), Image.ANTIALIAS)
	resized_image = np.array(resized_image.getdata())
	
	# FCNN takes in the flattened data
	return resized_image.flatten().astype(np.float64)
	
	
# Given an image path, return the feature vector for this image for the convolutional neural network
def process_image_CNN(image_path):
	# Read Image
	image = Image.open(image_path)
	
	# Resize the Image and convert to numpy array
	resized_image = image.resize((RES_WIDTH, RES_HEIGHT), Image.ANTIALIAS)
	resized_image = np.array(resized_image.getdata()).reshape(RES_WIDTH,RES_HEIGHT,3)
	
	# CNN takes in the data as a tensor: (samples, color channels, height, width)
	# so the color channels need to be brought to the front
	M, N, C = resized_image.shape
	
	return resized_image.reshape(C, M, N).astype(np.float64)
	
	
# Given a testing dataset and a classifier, classify each sample and determine the overall error rate
def test_sample(model, sample_image, process_function):
	data = process_function(sample_image)
	X = [data]
	X_test = Variable(torch.Tensor(X))
	
	pred_Y_test = model(X_test)
	prediction = list(pred_Y_test.data)
	
	return MODEL_CLASSES[prediction.index(max(prediction))]	

	
# Given a Base64 encoded image and an output path, produce the image file
def create_image(img_string, output_path):
	imgdata = base64.b64decode(img_string)
	with open(output_path, 'wb') as f:
		f.write(imgdata)


def get_scene(event, context):
	try:
		# Connect to S3 Bucket
		s3 = boto3.resource('s3')
		client_s3 = boto3.client('s3')
		bucket = s3.Bucket("altimit-test-bucket")
		
		# Check for Headers
		if ("mode" not in event) or ("imageBase64" not in event):
			return "Provide both the mode and imageBase64 headers"
			
		# Extract Headers
		mode = event["mode"].upper()
		image_string = event["imageBase64"]
		
		# Create working directory
		working_directory = "/tmp/scene-classifier"
		os.makedirs(working_directory, exist_ok=True)
		
		# Convert the sample image from the Base64 string
		sample_image_path = working_directory + "/sample_image.jpeg"
		create_image(image_string, sample_image_path)
			
		# Download the pre-trained model from S3
		model_path = working_directory + "/model_{}".format(mode.upper())
		with open(model_path, 'wb') as f:
			 client_s3.download_fileobj("altimit-test-bucket", "model_{}".format(mode.upper()), f)

		# Initialize PyTorch model based on the specified mode
		if mode == "FCNN":
			model = Net()
		elif mode == "CNN":
			model = ConvNet()
		else:
			return "Please specify the mode as either FCNN or CNN"
			
		# Load the PyTorch model
		model.load_state_dict(torch.load(model_path))
		model.eval()
		
		# Evaluate the sample
		if mode == "FCNN":
			prediction = test_sample(model, sample_image_path, process_image_FCNN)
		elif mode == "CNN":
			prediction = test_sample(model, sample_image_path, process_image_CNN)
			
		# Return the image classification
		return "Image has been classified as {}".format(prediction)

	except BaseException as error:
		return str(error)
		
