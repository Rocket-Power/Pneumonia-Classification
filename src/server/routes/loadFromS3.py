import boto3
from routes.awsinfo import awsInfo
from keras.models import load_model

# AWS S3 connection info
def loadModel():
  s3 = boto3.resource(
  service_name= awsInfo['service'],
  region_name= awsInfo['region'],
  aws_access_key_id= awsInfo['access_key'],
  aws_secret_access_key= awsInfo['s_access_key'] 
  )

  #download file from S3 bucket 
  s3.Bucket(awsInfo['bucket']).download_file(Key=awsInfo['file'], Filename=awsInfo['output'])

  # load saved model
  model = load_model('./model/model.h5') 
  return model