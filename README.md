# Overview
Scene Classification using PyTorch with a serverless AWS deployment option. See the uploaded PDF for experimental results. 

## Required Libs and Environment
* Python 3.6
* PyTorch
* NumPy
* PIL

## AWS Services
* API Gateway
* Lambda
* S3
* CloudFormation
* CloudWatch
* IAM



## Serverless AWS Deployment

###### Inside the "AWS Deployment" folder you'll find the CloudFormation template, Lambda Function source code, custom Lambda Layers (NumPy, PIL), and "YOUR_BUCKET_NAME" for the required S3 contents.

1. Upload the contents of "YOUR_BUCKET_NAME" to your S3 bucket. This includes the Lambda function code and some pre-trained FCNN and CNN models. These models were trained with a batch size of 32 and 100 epochs, feel free generate more robust models using the local run method described at the bottom of the page.

2. Ensure you have access to Lambda Layers for PyTorch, NumPy, and PIL.  
 
    PyTorch was a public layer that I found and NumPy / PIL are custom ones I compiled myself. I provided the zips in the "Lambda Layers" directory so you can upload them yourselves. 
   
    Once you have uploaded the layers, update the CloudFormation template accordingly with each layers ARN.

		"Layers": [
			"arn:aws:lambda:us-east-2:934676248949:layer:pytorchv1-py36:2",
			"PIL_ARN_HERE",
			"NUMPY_ARN_HERE"],
            
            
3. Replace "YOUR-BUCKET-NAME" with the bucket of your choice within the CloudFormation template.

4. Replace "DefaultUser" with a username of your choice within the CloudFormation template.

5. Create a new stack in CloudFormation and upload the provided template and create with default settings.

6. Try it out by issuing a POST request to your new API Gateway endpoint titled "Get_Scene API". 

    I recommend using Postman with the configuration described below. For the mode - use either "FCNN" (Fully Connected Neural Network) or "CNN" (Convolutional Neural Network). Provide the image as a Base64 encoded string.

****POST Message Body****

        {
          "mode": String,
          "imageBase64": String
        }

****POST Authorization****

		Provide the Access Key and Secret Key associated to the username you specified in step 4.


## Running Locally

****Train Network and Classify Test Samples****

The src directory contains main.py which is used for training and testing FCNN/CNN models.

	python main.py <train_dir> <test_dir> <mode>
	
The mode can be either "FCNN" (Fully Connected Neural Network) or "CNN" (Convolutional Neural Network). The test and train directories must contain a subdirectory for each class. See the provided example datasets in the train and test folders respectively.

After the run is complete, the model will be evaluated and metrics displayed to the end user. A file will be generated in the same directory as main.py titled either "model_FCNN" or "model_CNN" depending on the type launched. This model can be used locally by setting "LOAD_MODEL" to true in main.py. The models can also be uploaded into your S3 bucket for the AWS deployment described above.
