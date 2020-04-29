import tensorflow as tf
import pandas as pd
import os.path
from elmo import build_model, LMDataGenerator, DATA_SET_DIR, MODELS_DIR, parameters as elmo_parameters


reviews = pd.read_csv('_snappfood.csv')
texts = reviews['commentText'].values[: 60000]
feelings = reviews['feeling'].values
feelings = tf.constant([0.0 if feeling == 'SAD' else 1.0 for feeling in feelings][: 60000])

g = LMDataGenerator(os.path.join(DATA_SET_DIR, elmo_parameters['train_dataset']),
                    os.path.join(DATA_SET_DIR, elmo_parameters['vocab']),
                    sentence_maxlen=elmo_parameters['sentence_maxlen'],
                    token_maxlen=elmo_parameters['token_maxlen'],
                    batch_size=elmo_parameters['batch_size'],
                    shuffle=elmo_parameters['shuffle'])

elmo = build_model()
elmo.load_weights(tf.train.latest_checkpoint(MODELS_DIR))
vectors = g.encode(texts)
output_vectors = elmo.predict(vectors)[0]
print(len(output_vectors))
output_vectors = tf.constant(output_vectors)


# second = tf.keras.Sequential([
#     # tf.keras.layers.Embedding(8192, 64),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(1)
# ])

second = tf.keras.Sequential([
    # tf.keras.layers.Embedding(8192, 64),
    # tf.keras.layers.GlobalMaxPool1D(),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64),
    tf.keras.layers.LeakyReLU(alpha=0.05),
    tf.keras.layers.Dense(1)
])


second.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer=tf.keras.optimizers.Adam(1e-4), metrics=['accuracy'])
second.fit(output_vectors, feelings, epochs=200, validation_split=0.2, batch_size=256)
second.summary()
# print(second.predict(output_vectors[0]))
