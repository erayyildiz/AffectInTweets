from keras.models import Model
from keras.layers import Embedding, wrappers, LSTM, Dense, Dropout
from keras.layers import Input
from keras.preprocessing import sequence
import cPickle as pickle
from word2vecReader import Word2Vec
import data_utils
import math
import numpy as np


TRAIN_FILE_PATH = 'data/sentiment140/training.1600000.processed.noemoticon.csv'
TEST_FILE_PATH = 'data/sentiment140/testdata.manual.2009.06.14.csv'
MAX_SEQUENCE_LENGTH = 30
EMBEDDING_DIM = 400


def build_model(embedding_dim, max_word_count, embedding_matrix, max_sequence_length):
    embedding_layer = Embedding(max_word_count + 1,
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=max_sequence_length,
                                trainable=True)

    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    embedded_sequences = Dropout(0.2)(embedded_sequences)
    lstm_out = wrappers.Bidirectional(LSTM(LSTM_OUTPUT_LENGTH, return_sequences=False))(embedded_sequences)
    lstm_out = Dropout(0.2)(lstm_out)
    output = Dense(2, activation='softmax')(lstm_out)
    model = Model(input=[sequence_input], output=output)
    return model


def save_model(model, word2id, path='resources/saved_models/model'):
    model.save_weights(path + '.weights')
    with open(path + '.word2id', 'wb') as pickle_file:
        pickle.dump(word2id, pickle_file)


def load_model(model, path='resources/saved_models/model'):
    model.load_weights(path + '.weights')
    with open(path + '.word2id', 'rb') as pickle_file:
        word2id = pickle.load(pickle_file)
    return model, word2id


def train(model, word2id, x_train, y_train, x_dev, y_dev, batch_size=32):
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print('Train...')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=5,
              validation_data=(x_dev, y_dev))
    save_model(model, word2id)


def create_embedding_matrix(word2id, w2vModel, embedding_dim):
    embedding_matrix = np.zeros((len(word2id) + 1, embedding_dim))
    for word, i in word2id.iteritems():
        if word in w2vModel:
            embedding_matrix[i] = w2vModel[word]
        else:
            embedding_matrix[i] = np.random.normal(0, 0.2, embedding_dim)
    return embedding_matrix


if __name__ == "__main__":

    #TRAIN AND SAVE SENTIMENT MODEL
    print('Loading data...')
    x, y, word2id = data_utils.load_sentiment_data(TRAIN_FILE_PATH, frac=1)
    train_size = int(math.floor(0.9 * len(x)))
    x_train = x[:train_size]
    x_dev = x[train_size:]
    y_train = y[:train_size]
    y_dev = y[train_size:]
    print(len(x_train), 'train sequences')
    print(len(x_dev), 'test sequences')
    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=MAX_SEQUENCE_LENGTH)
    x_dev = sequence.pad_sequences(x_dev, maxlen=MAX_SEQUENCE_LENGTH)
    print('x_train shape:', x_train.shape)
    print('x_dev shape:', x_dev.shape)
    word2vec = Word2Vec.load_word2vec_format("resources/word2vec_twitter_model.bin", binary=True, word2id=word2id)
    pickle.dump(word2vec, "resources/twitter_word2vec.dic")
    embedding_matrix = create_embedding_matrix(word2id, word2vec, EMBEDDING_DIM)
    model = build_model(EMBEDDING_DIM, len(word2id), embedding_matrix, MAX_SEQUENCE_LENGTH)
    train(model, word2id, x_train, y_train, x_dev, y_dev, batch_size=8)

