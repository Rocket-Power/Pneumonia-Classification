from flask import Flask, request, render_template
from routes.loadFromS3 import loadModel
from routes.predict import makePrediction
from werkzeug.utils import secure_filename
import os

# declare a flask app initialized to a Flask instance
app = Flask(__name__)

# file path and allowed extensions for posting image for classification
UPLOAD_FOLDER = './image/'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', }

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# variable to store model
# model = loadModel()
from keras.models import load_model
model = load_model('./model/model.h5') 


# define an endpoint to handle predictions
@app.route("/predict", methods=["POST"])
def predict():
  print('INCOMING IMAGE!')
  file = request.files['file']
  
  #
  file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'image.jpeg'))

  prediction = makePrediction(model)

  if prediction[0][0] >= 0.5:
    print('You might have Pneumonia')
  else:
    print('You probably do not have Pneumonia')
  print("Prediction: ", prediction)
  return 200


if __name__ == "__main__":
  # run() method of Flask class runs the application on the local development server
  app.run()
