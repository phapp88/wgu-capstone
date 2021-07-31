import streamlit as st
import numpy as np
import pandas as pd

st.title('My first app')

df = pd.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [10, 20, 30, 40]
})

map_data = pd.DataFrame(
  np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
  columns=['lat','lon']
)

st.map(map_data)

if st.checkbox('Show dataframe'):
  chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['a','b','c']
  )

  st.line_chart(chart_data)

option = st.sidebar.selectbox(
  'Which number do you like best?',
  df['first column']
)

st.write('You selected: ', option)

left_column, right_column = st.beta_columns(2)
pressed = left_column.button('Press me?')
if pressed:
  right_column.write('Woohoo!')

expander = st.beta_expander('FAQ')
expander.write('Here you could put in some really, really long explanation...')