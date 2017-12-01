import word_lstm_sentiment_model
import data_utils
import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Embedding, wrappers, LSTM, Dense, Dropout, Input, Concatenate
from keras import backend as K
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from lexicon_features import LexiconFeatureExtractor

def create_feature_extraction_model(sentiment_pretrained=True, max_sequence_length=30, embedding_dim=300, word_index=None):
    word2id = word_index
    new_model = Sequential()
    if sentiment_pretrained:
        model, word2id, embedding_matrix = word_lstm_sentiment_model.load_model()
        for i in range(0, len(model.layers) - 1):
            new_model.add(model.layers[i])
            new_model.layers[-1].trainable = False
    else:
        embedding_layer = Embedding(len(word_index) + 1,
                                    embedding_dim,
                                    input_length=max_sequence_length,
                                    trainable=True)
        new_model.add(embedding_layer)
        new_model.add(Dropout(0.2))
        new_model.add(wrappers.Bidirectional(LSTM(embedding_dim, return_sequences=False)))
        new_model.add(Dropout(0.2))
    print new_model.summary()
    return new_model, word2id


def create_prediction_model(neural_features_len, lexicon_features_len, num_class):
    if lexicon_features_len > 0 and neural_features_len > 0 :
        lexicon_inputs = Input(shape=(lexicon_features_len,), name="lexicon_features")
        neural_inputs = Input(shape=(neural_features_len,), name="neural_features")
        neural_inputsd = Dense(128, activation="relu")(neural_inputs)
        merged_layer = Concatenate()([neural_inputsd, lexicon_inputs])
        regression_output = Dense(1, activation="sigmoid", name="regression_output")(merged_layer)
        classification_output = Dense(num_class, activation="softmax", name="classification_output")(merged_layer)
        prediction_model = Model(inputs=[neural_inputs, lexicon_inputs],
                                 outputs=[regression_output, classification_output])
    elif neural_features_len > 0 :
        neural_inputs = Input(shape=(lexicon_features_len,), name="lexicon_features")
        merged_layer = Dense(128, activation="relu")(neural_inputs)
        regression_output = Dense(1, activation="sigmoid", name="regression_output")(merged_layer)
        classification_output = Dense(num_class, activation="softmax", name="classification_output")(merged_layer)
        prediction_model = Model(inputs=neural_inputs,
                                 outputs=[regression_output, classification_output])
    elif lexicon_features_len > 0:
        lexicon_inputs = Input(shape=(lexicon_features_len,), name="neural_features")
        merged_layer = Dense(128, activation="relu")(lexicon_inputs)
        regression_output = Dense(1, activation="sigmoid", name="regression_output")(merged_layer)
        classification_output = Dense(num_class, activation="softmax", name="classification_output")(merged_layer)
        prediction_model = Model(inputs=lexicon_inputs,
                                 outputs=[regression_output, classification_output])
    else:
        raise Exception("One of the feature lengths must be greater than 0 at least!")




    prediction_model.compile(loss={"regression_output": "mean_squared_error",
                                   "classification_output": "categorical_crossentropy"},
                  optimizer='adam',
                  metrics={"regression_output": pearson_correlation_f, "classification_output": 'accuracy'})
    return prediction_model


def load_sentiment_model():
    print('Loading sentiment model...')
    model, word2id, embedding_matrix = word_lstm_sentiment_model.load_model()
    new_model = Sequential()
    for i in range(0, len(model.layers)-1):
        new_model.add(model.layers[i])
        new_model.layers[-1].trainable = False
    new_model.trainable = False
    new_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    return new_model, word2id, embedding_matrix


def extract_lexicon_features(tweet, lexicon_feature_extractor):
    return lexicon_feature_extractor.extract_feature(tweet)


def pearson_correlation_f(y_true, y_pred):
    fsp = y_pred - K.mean(y_pred, axis=0)  # you take the mean over the batch, keeping the features separate.
    fst = y_true - K.mean(y_true, axis=0)
    # mean shape: (1,10)
    # fst shape keeps (batch,10)

    devP = K.std(y_pred, axis=0)
    devT= K.std(y_true, axis=0)
    # dev shape: (1,10)

    return K.sum(K.mean(fsp * fst, axis=0) / (devP * devT))


def encode(arr, word2id):
    res = []
    for x in arr:
        res.append(data_utils.encode_sentence(x, word2id, None, threshold=0, add_unknowns=True))
    x = sequence.pad_sequences(res, maxlen=word_lstm_sentiment_model.MAX_SEQUENCE_LENGTH)
    return x


def train_model(name, regression_train_data_path, classification_train_data_path,
                regression_dev_data_path, classification_train_dev_path,
                word_index, use_neural_features=True, use_lexicon_features=True, feature_extraction_model=None):
    early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    lexicon_feature_extractor = LexiconFeatureExtractor()
    print "Preparing Data for {} model ...".format(name)
    train_x, train_y_labels = data_utils.load_affect_data(classification_train_data_path, is_label_numeric=False)
    _, train_y_scores = data_utils.load_affect_data(regression_train_data_path)
    train_y_labels = to_categorical(train_y_labels)
    train_y_scores = np.array(train_y_scores)
    dev_x, dev_y_labels = data_utils.load_affect_data(classification_train_dev_path, is_label_numeric=False)
    _, dev_y_scores = data_utils.load_affect_data(regression_dev_data_path)
    dev_y_scores = np.array(dev_y_scores)
    dev_y_labels = to_categorical(dev_y_labels)
    if use_neural_features and use_lexicon_features:
        neural_fetures_train = feature_extraction_model.predict(encode(train_x, word_index))
        lexicon_features_train = [extract_lexicon_features(instance, lexicon_feature_extractor) for instance in train_x]
        lexicon_features_train = np.array(lexicon_features_train)
        neural_fetures_dev = feature_extraction_model.predict(encode(dev_x, word_index))
        lexicon_features_dev = [extract_lexicon_features(instance, lexicon_feature_extractor) for instance in dev_x]
        lexicon_features_dev = np.array(lexicon_features_dev)
        model = create_prediction_model(len(neural_fetures_train[0]), len(lexicon_features_train[0]), len(train_y_labels[0]))
        print model.summary()
        print "Start training for {} model ...".format(name)
        model.fit(x=[neural_fetures_train, lexicon_features_train],
                  y={"regression_output": train_y_scores, "classification_output": train_y_labels},
                  validation_data=(
                      [neural_fetures_dev, lexicon_features_dev],
                      {"regression_output": dev_y_scores, "classification_output": dev_y_labels}
                  ),
                  epochs=100, callbacks=[early_stop], batch_size=1)
    elif use_lexicon_features and not use_neural_features:
        lexicon_features_train = [extract_lexicon_features(instance, lexicon_feature_extractor) for instance in train_x]
        lexicon_features_train = np.array(lexicon_features_train)
        lexicon_features_dev = [extract_lexicon_features(instance, lexicon_feature_extractor) for instance in dev_x]
        lexicon_features_dev = np.array(lexicon_features_dev)
        model = create_prediction_model(0, len(lexicon_features_train[0]), len(train_y_labels[0]))
        print model.summary()
        print "Start training for {} model ...".format(name)
        model.fit(x=lexicon_features_train,
                  y={"regression_output": train_y_scores, "classification_output": train_y_labels},
                  validation_data=(
                      lexicon_features_dev,
                      {"regression_output": dev_y_scores, "classification_output": dev_y_labels}
                  ),
                  epochs=100, callbacks=[early_stop], batch_size=1)
    elif not use_lexicon_features and use_neural_features:
        neural_fetures_train = feature_extraction_model.predict(encode(train_x, word_index))
        neural_fetures_dev = feature_extraction_model.predict(encode(dev_x, word_index))
        model = create_prediction_model(len(neural_fetures_train[0]), 0, len(train_y_labels[0]))
        print model.summary()
        print "Start training for {} model ...".format(name)
        model.fit(x=neural_fetures_train,
                  y={"regression_output": train_y_scores, "classification_output": train_y_labels},
                  validation_data=(
                      neural_fetures_dev,
                      {"regression_output": dev_y_scores, "classification_output": dev_y_labels}
                  ),
                  epochs=100, callbacks=[early_stop], batch_size=1)
    else:
        raise Exception("You must use one of the feature sets at least!")
    print "Saving {} model ...".format(name)
    model.save("resources/saved_models/{}.model".format(name))


if __name__ == "__main__":
    print "Creating Feature Extraction model"
    feature_extraction_model, word_index = create_feature_extraction_model()
    # feature_extraction_model = None
    # word_index = None
    train_model("joy_en", "data/semeval2018 task 1/EI-reg-en_joy_train.txt", "data/semeval2018 task 1/EI-oc-En-joy-train.txt",
                "data/semeval2018 task 1/2018-EI-reg-En-joy-dev.txt", "data/semeval2018 task 1/2018-EI-oc-En-joy-dev.txt",
                word_index, use_neural_features=True, use_lexicon_features=True,
                feature_extraction_model=feature_extraction_model)
