import os
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from bert import BertModelLayer
from bert.tokenization.bert_tokenization import FullTokenizer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights


# wget https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-4_H-512_A-8.zip
# run: ./q09-dowloading-bert.sh

BERT_MODEL_NAME = "uncased_L-4_H-512_A-8"  # BERT_MODEL_NAME="uncased_L-12_H-768_A-12"
BERT_CHECKPOINT_DIR = os.path.join("model/", BERT_MODEL_NAME)
BERT_CHECKPOINT_FILE = os.path.join(BERT_CHECKPOINT_DIR, "bert_model.ckpt")
BERT_CONFIG_FILE = os.path.join(BERT_CHECKPOINT_DIR, "bert_config.json")
RANDOM_SEED = 77
tokenizer = FullTokenizer(vocab_file=os.path.join(BERT_CHECKPOINT_DIR, "vocab.txt"))


class IntentRecognitionData:
    PARAM_COLUMN = "question"
    LABEL_COLUMN = "intent"

    def __init__(self, train, test, tokenizer: FullTokenizer, classes, maximum_sequence_length=50):
        self.tokenizer = tokenizer
        self.maximum_sequence_length = 0
        self.classes = classes
        ((self.train_x, self.train_y), (self.test_x, self.test_y)) = map(self._prepare, [train, test])
        self.maximum_sequence_length = min(self.maximum_sequence_length, maximum_sequence_length)
        self.train_x, self.test_x = map(self._pad, [self.train_x, self.test_x])
        np.random.seed(RANDOM_SEED)
        # print("max seq_len", self.maximum_sequence_length)

    def _prepare(self, df):
        x, y = [], []
        for _, row in tqdm(df.iterrows()):
            question, label = row[self.PARAM_COLUMN], row[self.LABEL_COLUMN]
            tokens = self.tokenizer.tokenize(question)
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            self.maximum_sequence_length = max(self.maximum_sequence_length, len(token_ids))
            x.append(token_ids)
            y.append(self.classes.index(label))
        return np.array(x), np.array(y)

    def _pad(self, ids):
        x = []
        for input_ids in ids:
            input_ids = input_ids[:min(len(input_ids), self.maximum_sequence_length - 2)]
            input_ids = input_ids + [0] * (self.maximum_sequence_length - len(input_ids))
            x.append(np.array(input_ids))
        return np.array(x)


def create_bert_model(maximum_sequence_length, bert_ckpt_file, classes):
    tf.random.set_seed(RANDOM_SEED)

    with tf.io.gfile.GFile(BERT_CONFIG_FILE, "r") as reader:
        bert_config = StockBertConfig.from_json_string(reader.read())
        bert_params = map_stock_config_to_params(bert_config)
        bert_params.adapter_size = None
        bert = BertModelLayer.from_params(bert_params, name="bert")
        
    input_ids = keras.layers.Input(shape=(maximum_sequence_length, ), dtype='int32', name="input_ids")
    bert_output = bert(input_ids)
    clases_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)
    clases_out = keras.layers.Dropout(0.5)(clases_out)
    logits = keras.layers.Dense(units=512, activation="tanh")(clases_out)
    logits = keras.layers.Dropout(0.5)(logits)
    logits = keras.layers.Dense(units=len(classes), activation="softmax")(logits)
    model = keras.Model(inputs=input_ids, outputs=logits)
    model.build(input_shape=(None, maximum_sequence_length))
    load_stock_weights(bert, bert_ckpt_file)
    # print("bert shape", bert_output.shape)
    return model

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
# print(train.shape)
# print(test.head())

classes = train.intent.unique().tolist()
# print(classes)
dataset = IntentRecognitionData(train, test, tokenizer, classes, maximum_sequence_length=50)
# print(dataset.train_x.shape)
# print(dataset.train_x[0])
# print(dataset.train_y[0])

model = create_bert_model(dataset.maximum_sequence_length, BERT_CHECKPOINT_FILE, classes)
model.summary()
    # Model: "model"
    # _________________________________________________________________
    # Layer (type)                 Output Shape              Param #
    # =================================================================
    # input_ids (InputLayer)       [(None, 25)]              0
    # _________________________________________________________________
    # bert (BertModelLayer)        (None, 25, 512)           28499968
    # _________________________________________________________________
    # lambda (Lambda)              (None, 512)               0
    # _________________________________________________________________
    # dropout (Dropout)            (None, 512)               0
    # _________________________________________________________________
    # dense (Dense)                (None, 512)               262656
    # _________________________________________________________________
    # dropout_1 (Dropout)          (None, 512)               0
    # _________________________________________________________________
    # dense_1 (Dense)              (None, 2)                 1026
    # =================================================================
    # Total params: 28,763,650
    # Trainable params: 28,763,650
    # Non-trainable params: 0
    # _________________________________________________________________

model.compile(
    optimizer=keras.optimizers.Adam(1e-7),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")]
)
log_dir = "log/intent_detection/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%s")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

history = model.fit(
    x=dataset.train_x, 
    y=dataset.train_y,
    validation_split=0.2,
    batch_size=4,
    shuffle=True,
    epochs=2,
    # callbacks=[tensorboard_callback]
)

loss, accuracy = model.evaluate(dataset.test)
print("Loss: ", loss)
print("Accuracy: ", accuracy)

model.save("question_intent_recognition_model.h5")



# maximum_sequence_length=550
# what was the last time Dany has been on the mainland of Westeros not on an adiacent island what was the last time Dany has been on the mainland of Essos not on an adiacent island what was the last time Dany has been on the mainland of Sothorios not on an adiacent island what was the last time Dany has been beyond the Wall not on an adiacent island what was the last time Dany has been on the Valyrian peninsula not on an adiacent island what was the last time Dany has been on the Dothraki sea not on an adiacent island what was the last time Dany has been on the Dragonstone not on an adiacent island what was the last time Dany has been on Bravos not on an adiacent island what was the last time Dany has been on Slavers' Bay not on an adiacent island what was the last time Dany has been in the house of the unding not on an adiacent island what was the last time Dany has been in King's Landing not on an adiacent island what was the last time Dany has been in Pentos not on an adiacent island what was the last time Dany has been in Oneros not on an adiacent island what was the last time Dany has been in Mereen not on an adiacent island what was the last time Dany has been in Astapor not on an adiacent island what was the last time Dany has been on the mainland of Westeros what was the last time Dany has been on the mainland of Westeros not on an adiacent island what was the last time Dany has been on the mainland of Westeros not on an adiacent island what was the last time Dany has been on the mainland of Westeros not on an adiacent island what was the last time Dany has been on the Iron islands not on an adiacent island what was the last time Dany has been on the Bear island not on an adiacent island what was the last time Dany has been in Winterfell not on an adiacent island what was the last time Dany has been in Ashay not on an adiacent island what was the last time Dany has been in the Basilisk ilands not on an adiacent island what was the last time Dany has been on the island of Naath not on an adiacent island?,AsoiafIntent

# InvalidArgumentError: Condition x <= y did not hold element-wise:
#     x (bert/embeddings/Const:0) =
#     518
#     y (bert/embeddings/position_embeddings/assert_less_equal/y:0) =
#     512