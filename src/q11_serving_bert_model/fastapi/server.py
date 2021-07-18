from fastapi import FastAPI, Request, Body, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import tensorflow as tf
from tensorflow import keras
import bert
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer
import os
import numpy as np

app = FastAPI(title='MLFAST',
              description='ML with FastAPI and Streamlit', version='0.0.1')
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

bert_model_name = "uncased_L-4_H-512_A-8"
bert_ckpt_dir = os.path.join("model/", bert_model_name)
tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, "vocab.txt"))
model = keras.models.load_model(
    "question_intent_recognition_model.h5", custom_objects={"BertModelLayer": BertModelLayer}
)
classes = [
    "BertIntent",
    "AsoiafIntent",
]


def get_ml_prediction(passage):
    pred_tok = tokenizer.tokenize(passage)
    pred_sep = ["[CLS]"] + pred_tok + ["[SEP]"]
    pred_ids = tokenizer.convert_tokens_to_ids(pred_sep)
    pred_pad = pred_ids + [0] * (25 - len(pred_ids))
    pred_num = np.array([pred_pad])
    predictions = model.predict(pred_num).argmax(axis=-1)
    label = classes[predictions[0]]
    return label


@app.get('/')
async def home(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})


@app.post('/predict')
async def get_predict(payload: dict = Body(...)):
    """Get ML prediction"""
    passage = payload['passage']
    label = get_ml_prediction(passage)
    prediction = {"prediction": label}
    prediction.update(payload)
    return prediction


@app.post('/prediction')
async def get_prediction(request: Request, passage: str = Form(...)):
    """Get ML prediction"""
    label = get_ml_prediction(passage)
    prediction = {"prediction": label}
    prediction.update({'passage': passage, 'request': request})
    return templates.TemplateResponse('index.html', prediction)
