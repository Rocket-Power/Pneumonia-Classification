import streamlit as st
import requests
import os

print(os.getcwd())

def predictionPage():
  # title of page
  st.title("Pneumonia Classification")

  # upload file here
  # initialize file to None
  file = None
  file = st.file_uploader('Upload File With Chest X-Ray')

  if file != None:
    st.image(file, width=500)

  # if the predict button is clicked send a request to the server that returns the predicted outcome
  if st.button("Predict"):
    if file != None:
      print(file)
      url = "http://localhost:5000/predict"
      payload = "img"
      print("Predicting!")

      # content-type: 'multipart/form-data'
      files = {'file': file, 
      'Content-Type': 'multipart/form-data'}
      response = requests.post(url, files=files)

      try:
        resp = response.json()
        resp = float(resp)
        if resp >= 0.5:
          st.header("Pneumonia Detected")
        elif resp < 0.5:
          st.header("Pneumonia Not Detected")
      except Exception as e:
        st.text(f"Could Not process request because: {e}")


def loadModelDetails():
  st.header('Resnet 50 Model')
  # st.header('Details:')
  # st.text('Added one Dense Layer of 256 neurons')
  # st.text('Dropout set to 0.10')
  st.markdown(
    """
    Model Details:
    - Resnet50 Base Model 
    - Added one Dense Layer of 256 neurons
    - Dropout set to 0.10
    """
  )
  st.image('./images/Classification_Report.png')
  st.image('./images/Confusion_matrix.png')

section = st.sidebar.selectbox(
  'What would you like to do?',
  ('Make Prediction', 'View Model Details')
)

if section == 'Make Prediction':
  predictionPage()
elif section == 'View Model Details':
  loadModelDetails()
else: 
  # Display default 
  predictionPage()


@st.cache
def load_model():
  pass