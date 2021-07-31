import glob
import io
import math
import numpy as np
import plotly as py
import plotly.graph_objs as go
import plotly.express as px
import random
import streamlit as st

from keras.applications.xception import preprocess_input
from keras.models import Model, load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
from sklearn.cluster import KMeans
from SessionState import get_state


# RECREATE kaggle notebooks with inputs cleaned up

# pages:
#   - if state.image is not None:
#     show full-sized image w/ a button to make a prediction
#     if user clicks button, show preprocessed image and the prediction
#     if the image is from the test batch, show the ground truth
#   - "View Images"
#     show batches of 8 images
#     user can click on an image to view the full size
#   - "Upload Image"
#     user can upload an image
#   - Kmeans plot page
#     just show plot, don't show which images correspond to which clusters
#     figure out optimal number of dimensions and clusters

# @st.cache(ttl=600)
# def run_query(query):

def main():
  state = get_state()
  model = get_model()
  test_images = get_test_images()

  if state.image is not None:
    # user has chosen an image to analyze
    st.image(state.image)
    diagnosis = get_diagnosis(model, state.image)
    st.write('Diagnosis: ' + diagnosis)
    
    if st.button('Back to dashboard'):
      state.image = None

  else:
    page = st.sidebar.selectbox('Choose a page', ['Upload Image', 'View Images', 'PCA Scatterplot'])

    if page == 'Upload Image':
      uploaded_image = st.file_uploader("Choose a file", type = ['png', 'jpg', 'jpeg'])

      if uploaded_image is not None:
        state.image = Image.open(io.BytesIO(uploaded_image.getvalue()))
    
    elif page == 'View Images':
      images = random.sample(test_images, 8)

      num_columns = 4
      num_rows = math.ceil(len(images) / num_columns)

      for i in range(num_rows):
        columns = st.beta_columns(num_columns)

        for j in range(len(columns)):
          column = columns[j]

          with column:
            image = images[i * num_columns + j].resize((160, 160))
            st.image(image)

            if st.button('Get diagnosis', key = str((i, j))):
              state.image = image

    elif page == 'PCA Scatterplot':
      test_set = np.concatenate([preprocess_image(image) for image in test_images], axis = 0)

      feature_extractor = Model(inputs = model.inputs, outputs = model.layers[-2].output)
      features = feature_extractor.predict(test_set)

      n_clusters = 3
      kmeans = KMeans(n_clusters = n_clusters)
      kmeans.fit(features)

      layout = go.Layout(
        title = '<b>Cluster Visualization</b>',
        yaxis = { 'title': '<i>Y</i>' },
        xaxis = { 'title': '<i>X</i>' }
      )

      colors = ['red', 'green', 'blue']
      trace = [go.Scatter3d() for _ in range(n_clusters)]
      for i in range(n_clusters):
        my_members = (kmeans.labels_ == i)
        index = [h for h, g, in enumerate(my_members) if g]
        trace[i] = go.Scatter3d(
          x = features[my_members, 0],
          y = features[my_members, 1],
          z = features[my_members, 2],
          mode = 'markers',
          marker = { 'size': 2, 'color': colors[i] },
          hovertext = index,
          name = 'Cluster' + str(i)
        )
      
      fig = go.Figure(data = [trace[0], trace[1], trace[2]], layout = layout)

      st.plotly_chart(fig)



  state.sync()

@st.cache
def get_model():
  model = load_model('models/trained_top_layer_model.h5')
  model.load_weights('weights/train-top-layer/weights.21.hdf5')

  return model

@st.cache
def get_test_images():
  images = []

  for filename in glob.glob('xrays/test/*/*.jpeg'):
    image = Image.open(filename)
    images.append(image)
  
  return images 

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

def get_features(model, images):
  pass

if __name__ == '__main__':
  main()
