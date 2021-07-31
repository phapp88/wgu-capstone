import glob
import io
import numpy as np
import os
import psycopg2
import streamlit as st

from keras.applications.xception import preprocess_input
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image

@st.cache(allow_output_mutation=True, hash_funcs={'_thread.RLock': lambda _: None})
def init_connection():
  return psycopg2.connect(**st.secrets['postgres'])

def login(connection):
  with connection:
    with connection.cursor() as cursor:
      select_statement = 'SELECT * FROM users WHERE username = %s AND password = %s'
      params = (st.session_state.username, st.session_state.password)
      cursor.execute(select_statement, params)
      results = cursor.fetchall()

      if len(results) == 1:
        st.session_state.is_logged_in = True

      else:
        st.session_state.login_message = 'The username or password you entered is incorrect.'

def register(connection):
  with connection:
    with connection.cursor() as cursor:
      try:
        insert_statement = 'INSERT INTO users (username, password) VALUES (%s, %s);'
        params = (st.session_state.username, st.session_state.password)
        cursor.execute(insert_statement, params)
        st.session_state.login_message = 'Registration Successful. You can now sign in.'
      
      except psycopg2.errors.UniqueViolation:
        st.session_state.login_message = 'That username already exists.'

@st.cache
def get_test_images():
  images = []

  for filename in glob.glob('xrays/test/*/*.jpeg'):
    image = Image.open(filename)
    images.append(image)
  
  return images 

@st.cache
def get_model():
  model = load_model('models/trained_top_layer_model.h5')
  model.load_weights('weights/train-top-layer/weights.21.hdf5')

  return model

def preprocess_image(image):
  if image.mode != 'RGB':
    image = image.convert('RGB')
  
  image = image.resize((299, 299))
  image = img_to_array(image)
  image = np.expand_dims(image, axis = 0)
  image = preprocess_input(image)

  return image

def get_diagnosis(model, image):
  preprocessed_image = preprocess_image(image)
  prediction = model.predict(preprocessed_image)[0][0]
  diagnosis = 'Normal' if prediction < 0.5 else 'Pneumonia'

  return diagnosis

def diagnosis_page(model):
  st.image(st.session_state.image_to_diagnose)
  diagnosis = get_diagnosis(model, st.session_state.image_to_diagnose)
  st.write('Diagnosis: ' + diagnosis)
  
  if st.button('Back to dashboard'):
    st.session_state.image_to_diagnose = None

def login_page(connection):
  with st.form(key='login_form', clear_on_submit=True):
      st.header('Login')

      st.text_input(label='Username', key='username')
      st.text_input(label='Password', type='password', key='password')

      st.write(st.session_state.login_message)

      st.form_submit_button(label='Sign In', on_click=login, args=(connection, ))
      st.form_submit_button(label='Register', on_click=register, args=(connection, ))

def upload_image_page():
  st.header('Upload an Image')

  uploaded_file = st.file_uploader(label='Choose a file', type = ['png', 'jpg', 'jpeg'])

  if uploaded_file is not None:
    uploaded_image = Image.open(io.BytesIO(uploaded_file.getvalue()))

    uploaded_image.save(f'xrays/{uploaded_file.name}.jpeg')

    st.session_state.image_to_diagnose = uploaded_image

def main():
  print('refresh')
  print(st.session_state)
  # put these variables in state?
  connection = init_connection()
  model = get_model()
  images = get_test_images()

  if not st.session_state:
    st.session_state.is_logged_in = False
    st.session_state.login_message = ''
    st.session_state.image_to_diagnose = None

  if not st.session_state.is_logged_in:
    login_page(connection)
  
  elif st.session_state.image_to_diagnose is not None:
    diagnosis_page(model)
  
  else:
    dashboard_page = st.sidebar.selectbox('Choose a page', ['Upload Image', 'View Images', 'PCA Scatterplot'])

    if dashboard_page == 'Upload Image':
      upload_image_page()
  
if __name__ == '__main__':
  main()
