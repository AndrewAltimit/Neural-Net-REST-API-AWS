# Overview
Image classification using a REST API and PyTorch/NumPy/PIL libraries. Includes AWS deployment utilizing CloudFormation, API Gateway, Lambda, and S3.

## Contributors

<a href="https://github.com/AndrewAltimit/Scene-Classification-AWS-Serverless/graphs/contributors">
  <img src="https://contributors-img.firebaseapp.com/image?repo=AndrewAltimit/Scene-Classification-AWS-Serverless" />
</a>

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



## AWS Deployment with CloudFormation

###### Inside the "AWS Deployment" folder you'll find the CloudFormation template, Lambda Function source code, custom Lambda Layers (NumPy, PIL), and the required S3 bucket contents.

1. Upload the files in **Bucket Contents** to your S3 bucket. This includes the Lambda function code and some pre-trained FCNN and CNN models. These models were lightly trained with a batch size = 32 and epochs = 100, feel free generate more robust models using the local run method described at the bottom of the page. 

2. Create a new stack in CloudFormation and upload **template.json**.

3. Specify a new IAM username, your bucket name, and Lambda Layer ARNs for PyTorch, NumPy, and PIL.
 
    PyTorch was a public layer that I found and NumPy / PIL are custom ones I compiled myself. I provided the zips in the **Lambda Layers** directory so you can upload them yourselves if needed.
   

4. Once the deployment is finished you can try it out by issuing a POST request to your new API Gateway endpoint titled "Get_Scene API". 

    I recommend using Postman with the configuration described below. Include the mode - either "FCNN" (Fully Connected Neural Network) or "CNN" (Convolutional Neural Network), the source model bucket name, and the sample image as a Base64 encoded string.

****POST Message Body****

        {
          "mode": String,
          "imageBase64": String,
          "bucketName": String
        }

****POST Authorization****

		Provide the Access Key and Secret Key associated to the username you specified in step 3.


## Running Locally

****Train Network and Classify Test Samples****

The src directory contains main.py which is used for training and testing FCNN/CNN models.

	python main.py <train_dir> <test_dir> <mode>
	
The mode can be either "FCNN" (Fully Connected Neural Network) or "CNN" (Convolutional Neural Network). The test and train directories must contain a subdirectory for each class. See the example dataset by extracting dataset.zip (split across dataset.zip.001 and dataset.zip.002).

After the run is complete, the model will be evaluated and metrics displayed to the end user. A file will be generated in the same directory as main.py titled either "model_FCNN" or "model_CNN" depending on the type launched. This model can be used locally by setting "LOAD_MODEL" to true in main.py. The models can also be uploaded into your S3 bucket for the AWS deployment described above.

