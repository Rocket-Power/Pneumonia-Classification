import streamlit as st
import requests

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
    print("File: ")
    print(file)
    url = "url"
    payload = "img"
    print("Predicting!")
    # response = requests.post(url, json = {'img': payload})

    # try:
    #   resp = response.json()
    #   if resp == 1:
    #     st.text("Pneumonia Detected")
    #   elif resp == 0:
    #     st.text("Pneumonia Not Detected")
    # except Exception as e:
    #   st.text(f"Could Not process request because: {e}")


st.sidebar.selectbox(
  'How would you like your prediction results?',
  ('Email', 'Text')
)


@st.cache
def load_model():
  pass