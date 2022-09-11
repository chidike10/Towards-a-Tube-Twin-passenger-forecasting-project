import streamlit as st

header = st.container()
data1 = st.container()
features = st.container()
model_training = st.container()

with header:
	st.title('Title 1')
	st.text('Text 1')


with data1:
	st.header('Header 1')
	st.text('Text 2')


with features:
	st.header('Header 2')
	st.markdown('Mark down 1')
	st.markdown('Mark down 2')


with model_training:
	st.header('Header 3')
