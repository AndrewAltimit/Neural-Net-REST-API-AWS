# Overview
Scene Classification using PyTorch. See the uploaded PDF for experimental results. 

If you want the image datasets used in the paper, please contact me by email: andrew.altimit@gmail.com

## Prerequisites
* Python 3.* https://www.python.org/download/
* PyTorch
* Numpy
* OpenCV 3.*
* sklearn (datasets and metrics)
 

## Usage

***Train Network and Classify Test Samples***

	python main.py <train_dir> <test_dir> <mode>
	
The mode can be either "FCNN" (Fully Connected Neural Network) or "CNN" (Convolutional Neural Network). The test and train directories must contain a subdirectory for each class. 
	
