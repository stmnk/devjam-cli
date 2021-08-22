import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import layers

maximum_sequence_length = 384
input_shape = (maximum_sequence_length,)

input_word_ids = tf.keras.layers.Input(
    shape=input_shape, 
    dtype=tf.int32, 
    name='input_word_ids'
)
input_mask     = tf.keras.layers.Input(
    shape=input_shape, 
    dtype=tf.int32, 
    name='input_mask'
)
input_type_ids = tf.keras.layers.Input(
    shape=input_shape, 
    dtype=tf.int32, 
    name='input_type_ids'
)

qa_model_inputs = [input_word_ids, input_mask, input_type_ids]

pooled_output, sequence_output = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2", 
    trainable=True
)(qa_model_inputs)


start_logits = layers.Dense(
    1, 
    name="start_logit", 
    use_bias=False
)(sequence_output)
start_logits = layers.Flatten()(start_logits)
start_probs = layers.Activation(keras.activations.softmax)(start_logits)

end_logits = layers.Dense(
    1, 
    name="end_logit", 
    use_bias=False
)(sequence_output)
end_logits = layers.Flatten()(end_logits)
end_probs = layers.Activation(keras.activations.softmax)(end_logits)

qa_model_outputs = [start_probs, end_probs]

model = keras.Model(
    inputs=qa_model_inputs, 
    outputs=qa_model_outputs
)

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
optimizer = keras.optimizers.Adam(
    learning_rate=1e-5, 
    beta_1=0.9, 
    beta_2=0.98, 
    epsilon=1e-9
)

model.compile(optimizer=optimizer, loss=[loss, loss])

model.summary()