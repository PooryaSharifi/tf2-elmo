import tensorflow as tf
from custom_layers import QRNN
import os
from pathlib import Path
import numpy as np
import sentencepiece as spm

# !pip install sentencepiece
# upload custom_layers

DATA_BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MODELS_DIR = os.path.join(DATA_BASE_DIR, 'elmo_checkpoints')
DATA_SET_DIR = os.path.join(DATA_BASE_DIR, 'elmo_datasets')
if not os.path.exists(DATA_SET_DIR):
    Path(DATA_SET_DIR).mkdir(parents=True, exist_ok=True)
    # Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    os.chdir(DATA_BASE_DIR)
    os.system("git clone https://github.com/kafura-kafiri/elmo_checkpoints.git")
    os.system('git config --global user.name "poorya"')
    os.system('git config --global user.email "kafura.kafiri@gmail.com"')
    os.system("git clone https://github.com/kafura-kafiri/data.git")
    os.chdir(os.path.join(DATA_BASE_DIR, 'data'))
    os.system("cat telegram_0 telegram_1 snappfood_1 snappfood_2 voa_0 voa_1 wikipedia_0 wikipedia_1 > corpus.txt")
    os.system("mv corpus.txt ../elmo_datasets")
    os.system("mv corpus.model ../elmo_datasets")

if not os.path.exists(os.path.join(MODELS_DIR, '.git')):
    os.chdir(MODELS_DIR)
    os.system('git init .')
    os.system('git config --global user.name "poorya"')
    os.system('git config --global user.email "kafura.kafiri@gmail.com"')
    os.system('git remote add origin https://github.com/kafura-kafiri/elmo_checkpoints.git')


checkpoint_prefix = os.path.join(MODELS_DIR, "ckpt_{epoch}")

parameters = {
    'train_dataset': 'corpus.txt',
    'vocab': "corpus.model",
    'vocab_size': 8192,
    'num_sampled': 1000,
    'sentence_maxlen': 64,
    'token_maxlen': 50,
    'token_encoding': 'word',
    'epochs': 64,
    'batch_size': 256,
    'clip_value': 1,
    'cell_clip': 5,
    'proj_clip': 5,
    'lr': 0.2,
    'shuffle': True,
    'n_lstm_layers': 2,
    'cnn_filters': [[1, 32],
                    [2, 32],
                    [3, 64],
                    [4, 128],
                    [5, 256],
                    [6, 512],
                    [7, 512]
                    ],
    'lstm_units_size': 300,
    'hidden_units_size': 200,
    'dropout_rate': 0.1,
    'word_dropout_rate': 0.05,
}


class LMDataGenerator(tf.keras.utils.Sequence):
    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __init__(self, corpus, vocab, sentence_maxlen=100, token_maxlen=50, batch_size=32, shuffle=True):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(vocab)
        self.corpus = corpus
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sentence_maxlen = sentence_maxlen
        self.token_maxlen = token_maxlen
        npy = corpus.split('.txt')[0] + f'_s{sentence_maxlen}.npy'
        if not os.path.exists(npy):
            lines = []
            with open(corpus, 'rU', encoding='utf-8') as f:
                for l in f.readlines():
                    lines.append(self.get_token_indices(l))
            np.save(npy, np.array(lines))

        self.vectors = np.load(npy)
        self.indices = np.arange(len(self.vectors))
        newlines = [index for index in range(0, len(self.indices), 2)]
        self.indices = np.delete(self.indices, newlines)

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        word_indices_batch = np.zeros((len(batch_indices), self.sentence_maxlen), dtype=np.int32)
        for i, batch_id in enumerate(batch_indices):
            word_indices_batch[i] = self.vectors[batch_id]

        # Build forward targets
        for_word_indices_batch = np.zeros((len(batch_indices), self.sentence_maxlen), dtype=np.int32)

        padding = np.zeros((1, ), dtype=np.int32)

        for i, word_seq in enumerate(word_indices_batch ):
            for_word_indices_batch[i] = np.concatenate((word_seq[1:], padding), axis=0)

        for_word_indices_batch = for_word_indices_batch[:, :, np.newaxis]
        # Build backward targets
        back_word_indices_batch = np.zeros((len(batch_indices), self.sentence_maxlen), dtype=np.int32)

        for i, word_seq in enumerate(word_indices_batch):
            back_word_indices_batch[i] = np.concatenate((padding, word_seq[:-1]), axis=0)

        back_word_indices_batch = back_word_indices_batch[:, :, np.newaxis]
        return [word_indices_batch, for_word_indices_batch, back_word_indices_batch], []

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def get_token_indices(self, line):
        ids = self.sp.EncodeAsIds(line)[: self.sentence_maxlen - 2]
        x = np.zeros((self.sentence_maxlen, ), dtype=np.int32)
        x[1: len(ids) + 1] = ids
        x[0] = 1
        x[len(ids) + 1] = 2
        return x

    def encode(self, ms):
        word_indices_batch = np.zeros((len(ms), self.sentence_maxlen), dtype=np.int32)
        for i, m in enumerate(ms):
            word_indices_batch[i] = self.get_token_indices(m)

        for_word_indices_batch = np.zeros((len(ms), self.sentence_maxlen), dtype=np.int32)

        padding = np.zeros((1,), dtype=np.int32)

        for i, word_seq in enumerate(word_indices_batch):
            for_word_indices_batch[i] = np.concatenate((word_seq[1:], padding), axis=0)

        for_word_indices_batch = for_word_indices_batch[:, :, np.newaxis]

        back_word_indices_batch = np.zeros((len(ms), self.sentence_maxlen), dtype=np.int32)

        for i, word_seq in enumerate(word_indices_batch):
            back_word_indices_batch[i] = np.concatenate((padding, word_seq[:-1]), axis=0)

        back_word_indices_batch = back_word_indices_batch[:, :, np.newaxis]
        return [word_indices_batch, for_word_indices_batch, back_word_indices_batch]


class TimestepDropout(tf.keras.layers.Dropout):
    def __init__(self, rate, **kwargs):
        super(TimestepDropout, self).__init__(rate, **kwargs)
        self.input_spec = tf.keras.layers.InputSpec(ndim=3)

    @staticmethod
    def _get_noise_shape(inputs):
        input_shape = tf.shape(inputs)
        noise_shape = (input_shape[0], input_shape[1], 1)
        return noise_shape


class Camouflage(tf.keras.layers.Layer):
    def __init__(self, mask_value=0., **kwargs):
        super(Camouflage, self).__init__(**kwargs)
        self.mask_value = mask_value

    def call(self, inputs):
        boolean_mask = tf.keras.backend.any(tf.keras.backend.not_equal(inputs[1], self.mask_value),
                                            axis=-1, keepdims=True)
        return inputs[0] * tf.keras.backend.cast(boolean_mask, tf.keras.backend.dtype(inputs[0]))

    def get_config(self):
        config = {'mask_value': self.mask_value}
        base_config = super(Camouflage, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class SampledSoftmax(tf.keras.layers.Layer):
    def __init__(self, num_classes=50000, num_sampled=1000, tied_to=None, **kwargs):
        super(SampledSoftmax, self).__init__(**kwargs)
        self.num_sampled = num_sampled
        self.num_classes = num_classes
        self.tied_to = tied_to
        self.sampled = (self.num_classes != self.num_sampled)

    def build(self, input_shape):
        if self.tied_to is None:
            self.softmax_W = self.add_weight(shape=(self.num_classes, input_shape[0][-1]), name='W_soft',
                                             initializer='lecun_normal')
        self.softmax_b = self.add_weight(shape=(self.num_classes,), name='b_soft', initializer='zeros')
        self.built = True

    def call(self, x, mask=None):
        lstm_outputs, next_token_ids = x

        def sampled_softmax(x):
            lstm_outputs_batch, next_token_ids_batch = x
            batch_losses = tf.nn.sampled_softmax_loss(
                self.softmax_W if self.tied_to is None else self.tied_to.weights[0], self.softmax_b,
                next_token_ids_batch, lstm_outputs_batch,
                num_classes=self.num_classes,
                num_sampled=self.num_sampled,
                # partition_strategy='div'
            )
            batch_losses = tf.reduce_mean(batch_losses)
            return [batch_losses, batch_losses]

        def softmax(x):
            lstm_outputs_batch, next_token_ids_batch = x
            logits = tf.matmul(lstm_outputs_batch,
                               tf.transpose(self.softmax_W) if self.tied_to is None else tf.transpose(
                                   self.tied_to.weights[0]))
            logits = tf.nn.bias_add(logits, self.softmax_b)
            batch_predictions = tf.nn.softmax(logits)
            labels_one_hot = tf.one_hot(tf.cast(next_token_ids_batch, dtype=tf.int32), self.num_classes)
            batch_losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels_one_hot, logits=logits)
            return [batch_losses, batch_predictions]

        losses, predictions = tf.map_fn(sampled_softmax if self.sampled else softmax, [lstm_outputs, next_token_ids])
        self.add_loss(0.5 * tf.reduce_mean(losses[0]))
        return lstm_outputs if self.sampled else predictions

    def compute_output_shape(self, input_shape):
        return input_shape[0] if self.sampled else (input_shape[0][0], input_shape[0][1], self.num_classes)


def build_model():
    reverse = lambda inputs: tf.keras.backend.reverse(inputs, axes=1)
    # Train word embeddings from scratch
    word_inputs = tf.keras.layers.Input(shape=(None,), name='word_indices', dtype='int32')
    embeddings = tf.keras.layers.Embedding(parameters['vocab_size'], parameters['hidden_units_size'], trainable=True,
                                           name='token_encoding')
    inputs = embeddings(word_inputs)
    drop_inputs = tf.keras.layers.SpatialDropout1D(parameters['dropout_rate'])(inputs)
    lstm_inputs = TimestepDropout(parameters['word_dropout_rate'])(drop_inputs)

    # Pass outputs as inputs to apply sampled softmax
    next_ids = tf.keras.layers.Input(shape=(None, 1), name='next_ids', dtype='float32')
    previous_ids = tf.keras.layers.Input(shape=(None, 1), name='previous_ids', dtype='float32')

    # Reversed input for backward LSTMs
    re_lstm_inputs = tf.keras.layers.Lambda(function=reverse)(lstm_inputs)
    mask = tf.keras.layers.Lambda(function=reverse)(drop_inputs)

    # Forward LSTMs
    for i in range(parameters['n_lstm_layers']):
        lstm = QRNN(units=parameters['lstm_units_size'], return_sequences=True, activation="tanh",
                                    recurrent_activation='sigmoid',
                                    kernel_constraint=tf.keras.constraints.MinMaxNorm(-1 * parameters['cell_clip'],
                                                                                      parameters['cell_clip']),
                                    recurrent_constraint=tf.keras.constraints.MinMaxNorm(-1 * parameters['cell_clip'],
                                                                                         parameters['cell_clip'])
                                    )(lstm_inputs)
        lstm = Camouflage(mask_value=0)(inputs=[lstm, drop_inputs])
        # Projection to hidden_units_size
        proj = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(parameters['hidden_units_size'], activation='linear',
                                  kernel_constraint=tf.keras.constraints.MinMaxNorm(-1 * parameters['proj_clip'],
                                                                                    parameters['proj_clip'])
                                  ))(lstm)
        # Merge Bi-LSTMs feature vectors with the previous ones
        lstm_inputs = tf.keras.layers.add([proj, lstm_inputs], name='f_block_{}'.format(i + 1))
        # Apply variational drop-out between BI-LSTM layers
        lstm_inputs = tf.keras.layers.SpatialDropout1D(parameters['dropout_rate'])(lstm_inputs)

    # Backward LSTMs
    for i in range(parameters['n_lstm_layers']):
        re_lstm = QRNN(units=parameters['lstm_units_size'], return_sequences=True, activation='tanh',
                                       recurrent_activation='sigmoid',
                                       kernel_constraint=tf.keras.constraints.MinMaxNorm(-1 * parameters['cell_clip'],
                                                                                         parameters['cell_clip']),
                                       recurrent_constraint=tf.keras.constraints.MinMaxNorm(
                                           -1 * parameters['cell_clip'],
                                           parameters['cell_clip'])
                                       )(re_lstm_inputs)
        re_lstm = Camouflage(mask_value=0)(inputs=[re_lstm, mask])
        # Projection to hidden_units_size
        re_proj = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(parameters['hidden_units_size'], activation='linear',
                                  kernel_constraint=tf.keras.constraints.MinMaxNorm(-1 * parameters['proj_clip'],
                                                                                    parameters['proj_clip'])
                                  ))(re_lstm)
        # Merge Bi-LSTMs feature vectors with the previous ones
        re_lstm_inputs = tf.keras.layers.add([re_proj, re_lstm_inputs], name='b_block_{}'.format(i + 1))
        # Apply variational drop-out between BI-LSTM layers
        re_lstm_inputs = tf.keras.layers.SpatialDropout1D(parameters['dropout_rate'])(re_lstm_inputs)

    # Reverse backward LSTMs' outputs = Make it forward again
    re_lstm_inputs = tf.keras.layers.Lambda(function=reverse, name="reverse")(re_lstm_inputs)

    # Project to Vocabulary with Sampled Softmax
    sampled_softmax = SampledSoftmax(num_classes=parameters['vocab_size'],
                                     num_sampled=int(parameters['num_sampled']),
                                     tied_to=embeddings)
    outputs = sampled_softmax([lstm_inputs, next_ids])
    re_outputs = sampled_softmax([re_lstm_inputs, previous_ids])

    m = tf.keras.models.Model(inputs=[word_inputs, next_ids, previous_ids], outputs=[outputs, re_outputs])
    m.compile(optimizer=tf.keras.optimizers.Adagrad(lr=parameters['lr'], clipvalue=parameters['clip_value']), loss=None, metrics=['accuracy'])
    return m


def train(_model, train_data):
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)

    def custom_callback(*args):
        os.chdir(MODELS_DIR)
        os.system('git add *')
        os.system('git commit -m "new_checkpoint"')
        os.system('git push "https://kafura-kafiri:Po00orya@github.com/kafura-kafiri/elmo_checkpoints.git" --all')
        print('did it just push')

    custom_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=custom_callback)
    _model.fit_generator(train_data, epochs=parameters['epochs'], callbacks=[checkpoint_callback, custom_callback])


model = build_model()
# model.summary()

g = LMDataGenerator(os.path.join(DATA_SET_DIR, parameters['train_dataset']),
                                  os.path.join(DATA_SET_DIR, parameters['vocab']),
                                  sentence_maxlen=parameters['sentence_maxlen'],
                                  token_maxlen=parameters['token_maxlen'],
                                  batch_size=parameters['batch_size'],
                                  shuffle=parameters['shuffle'])

try:
    model.load_weights(tf.train.latest_checkpoint(MODELS_DIR))
except: pass
train(model, g)
