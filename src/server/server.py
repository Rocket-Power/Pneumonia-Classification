from flask import Flask, request, Response
from routes.loadFromS3 import loadModel
from routes.predict import makePrediction
from werkzeug.utils import secure_filename
import os

# declare a flask app initialized to a Flask instance
app = Flask(__name__)

# file path and allowed extensions for posting image for classification
UPLOAD_FOLDER = './image/'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# variable to store model
model = loadModel()
# from keras.models import load_model
# model = load_model('./model/model.h5') 


# define an endpoint to handle predictions
@app.route("/predict", methods=["POST"])
def predict():
  # verify the content type from the request header
  content_type = request.headers.get('Content-type')
  content_type = content_type.split(';')[0]
  
  if content_type == 'multipart/form-data':
  
    # check to see if data was sent with the key - 'file'
    try:
      file = request.files['file']
    except Exception as e:
      return Response(response='Invalid No File', status=406)

    # validate the extension is within the parameters of our model
    extension = file.filename.split('.')[1]

    if extension in ALLOWED_EXTENSIONS:
      # save the file to image directory for use in predict.py
      file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'image.jpeg'))

      # return the model prediction on the requests image
      prediction = makePrediction(model)

      # verify a proper prediction was made and send the result back to the client
      # otherwise handle errors and send back appropriate error message / status code to client
      if prediction != None:
          return Response(response=str(prediction[0][0]), status=200)
      return Response(response='Invalid Prediction', status=401)
      
    return Response(response='Invalid Extension', status=415)

  return Response(response='Invalid Header Content-Type', status=412)


if __name__ == "__main__":
  # run() method of Flask class runs the application on the local development server
  app.run()
