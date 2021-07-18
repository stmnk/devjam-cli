import streamlit as st
import requests as req

endpoint = backend = 'http://fastapi:8000/predict'

def get_prediction(url, passage):
    headers = {'Content-Type': 'application/json'}
    res = req.post(url, json={'passage': passage}, headers=headers) 
    return res

st.title('ML FA ST')

st.write(
    'ML prediction; FastAPI [endpoint](http://localhost:8000/docs); Streamlit [frontend](http://localhost:8501)'
) 

default_value = 'how many languages does roberta understand?'
passage = st.text_input('Input an question to predict its epistemic intent: ', default_value)

if st.button('get prediction'):
    if passage:
        resp = get_prediction(endpoint, passage)
        # prediction = resp.content
        prediction = resp.json()
        st.write(prediction)
    else:
        st.write('Write some passage of text')