from keras.models import Model
from keras.layers import Embedding, wrappers, LSTM, Dense, Dropout
from keras.layers import Input
from keras.preprocessing import sequence
import cPickle as pickle
from word2vecReader import Word2Vec
import data_utils
import math


TRAIN_FILE_PATH = 'data/sentiment140/training.1600000.processed.noemoticon.csv'
TEST_FILE_PATH = 'data/sentiment140/testdata.manual.2009.06.14.csv'
MAX_SEQUENCE_LENGTH = 30

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


def save_model(model, path='resources/saved_models/model'):
    model.save_weights(path + '.weights')
    with open(path + '.word2id', 'wb') as pickle_file:
        pickle.dump(word2id, pickle_file)


def load_model(model, path='resources/saved_models/model'):
    model.load_weights(path + '.weights')
    with open(path + '.word2id', 'rb') as pickle_file:
        word2id = pickle.load(pickle_file)
    return model, word2id


def train(model, x_train, y_train, x_dev, y_dev, batch_size=32):
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print('Train...')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=5,
              validation_data=(x_dev, y_dev))
    save_model(model)


if __name__ == "__main__":
    print('Loading data...')
    x, y, word2id = data_utils.load_sentiment_data(TRAIN_FILE_PATH, frac=0.01)
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
    word2vec = Word2Vec.load_word2vec_format("resources/word2vec_twitter_model.bin", binary=True)
