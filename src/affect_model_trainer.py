import word_lstm_sentiment_model
import data_utils
import numpy as np
import cPickle as pickle
from keras.models import Sequential, Model, load_model
from keras.layers import Embedding, wrappers, LSTM, Dense, Dropout, Input, Concatenate
from keras import backend as K, metrics
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from lexicon_features import LexiconFeatureExtractor


def create_feature_extraction_model(word2id_path='resources/sentiment_model_word2id.dic',
                                    sentiment_model_path='resources/saved_models/sentiment.model'):
    new_model = Sequential()
    model = load_model(sentiment_model_path)
    with open(word2id_path, 'rb') as pickle_file:
        word2id = pickle.load(pickle_file)
    for i in range(0, len(model.layers) - 1):
        new_model.add(model.layers[i])
        new_model.layers[-1].trainable = False
    new_model.trainable = False
    print new_model.summary()

    return new_model, word2id


def create_prediction_model(sentiment_features_len, lexicon_features_len, num_class, vocab_size,
                            embedding_dim=50,
                            max_sequence_length=word_lstm_sentiment_model.MAX_SEQUENCE_LENGTH,
                            embedding_matrix=None):
    if embedding_matrix is not None:
        embedding_dim = word_lstm_sentiment_model.EMBEDDING_DIM
    if lexicon_features_len > 0 and sentiment_features_len > 0:
        word_inputs = Input(shape=(max_sequence_length,), name="word_features")
        if embedding_matrix is not None:
            embedding_layer = Embedding(vocab_size + 1, embedding_dim, trainable=True, weights=[embedding_matrix])
        else:
            embedding_layer = Embedding(vocab_size + 1, embedding_dim, trainable=True)
        word_embeddings = embedding_layer(word_inputs)
        word_embeddingsd = Dropout(0.2)(word_embeddings)
        word_lstm_outputs = wrappers.Bidirectional(LSTM(embedding_dim, return_sequences=False))(word_embeddingsd)
        word_lstm_outputsd = Dropout(0.2)(word_lstm_outputs)
        lexicon_inputs = Input(shape=(lexicon_features_len,), name="lexicon_features")
        sentiment_inputs = Input(shape=(sentiment_features_len,), name="sentiment_features")
        sentiment_inputsd = Dropout(0.2)(sentiment_inputs)
        sentiment_inputsdd = Dense(128, activation="relu")(sentiment_inputsd)
        merged_layer = Concatenate()([word_lstm_outputsd, sentiment_inputsdd, lexicon_inputs])
        merged_layerd = Dropout(0.2)(merged_layer)
        merged_layerdd = Dense(64, activation="relu")(merged_layerd)
        regression_output = Dense(1, activation="sigmoid", name="regression_output")(merged_layerdd)
        classification_output = Dense(num_class, activation="softmax", name="classification_output")(merged_layerdd)
        prediction_model = Model(inputs=[word_inputs, sentiment_inputs, lexicon_inputs],
                                 outputs=[regression_output, classification_output])
    elif sentiment_features_len > 0:
        word_inputs = Input(shape=(max_sequence_length,), name="word_features")
        if embedding_matrix is not None:
            embedding_layer = Embedding(vocab_size + 1, embedding_dim, trainable=True, weights=[embedding_matrix])
        else:
            embedding_layer = Embedding(vocab_size + 1, embedding_dim, trainable=True)
        word_embeddings = embedding_layer(word_inputs)
        word_embeddingsd = Dropout(0.2)(word_embeddings)
        word_lstm_outputs = wrappers.Bidirectional(LSTM(embedding_dim, return_sequences=False))(word_embeddingsd)
        word_lstm_outputsd = Dropout(0.2)(word_lstm_outputs)
        sentiment_inputs = Input(shape=(sentiment_features_len,), name="sentiment_features")
        sentiment_inputsd = Dropout(0.2)(sentiment_inputs)
        sentiment_inputsdd = Dense(128, activation="relu")(sentiment_inputsd)
        merged_layer = Concatenate()([word_lstm_outputsd, sentiment_inputsdd])
        merged_layerd = Dropout(0.2)(merged_layer)
        merged_layerdd = Dense(64, activation="relu")(merged_layerd)
        regression_output = Dense(1, activation="sigmoid", name="regression_output")(merged_layerdd)
        classification_output = Dense(num_class, activation="softmax", name="classification_output")(merged_layerdd)
        prediction_model = Model(inputs=[word_inputs, sentiment_inputs],
                                 outputs=[regression_output, classification_output])
    elif lexicon_features_len > 0:
        word_inputs = Input(shape=(max_sequence_length,), name="word_features")
        if embedding_matrix is not None:
            embedding_layer = Embedding(vocab_size + 1, embedding_dim, trainable=True, weights=[embedding_matrix])
        else:
            embedding_layer = Embedding(vocab_size + 1, embedding_dim, trainable=True)
        word_embeddings = embedding_layer(word_inputs)
        word_embeddingsd = Dropout(0.2)(word_embeddings)
        word_lstm_outputs = wrappers.Bidirectional(LSTM(embedding_dim, return_sequences=False))(word_embeddingsd)
        word_lstm_outputsd = Dropout(0.2)(word_lstm_outputs)
        lexicon_inputs = Input(shape=(lexicon_features_len,), name="sentiment_features")
        merged_layer = Concatenate()([word_lstm_outputsd, lexicon_inputs])
        merged_layerd = Dense(64, activation="relu")(merged_layer)
        regression_output = Dense(1, activation="sigmoid", name="regression_output")(merged_layerd)
        classification_output = Dense(num_class, activation="softmax", name="classification_output")(merged_layerd)
        prediction_model = Model(inputs=[word_inputs, lexicon_inputs],
                                 outputs=[regression_output, classification_output])
    else:
        word_inputs = Input(shape=(max_sequence_length,), name="word_features")
        if embedding_matrix is not None:
            embedding_layer = Embedding(vocab_size + 1, embedding_dim, trainable=True, weights=[embedding_matrix])
        else:
            embedding_layer = Embedding(vocab_size + 1, embedding_dim, trainable=True)
        word_embeddings = embedding_layer(word_inputs)
        word_embeddingsd = Dropout(0.2)(word_embeddings)
        word_lstm_outputs = wrappers.Bidirectional(LSTM(embedding_dim, return_sequences=False))(word_embeddingsd)
        word_lstm_outputsd = Dropout(0.2)(word_lstm_outputs)
        merged_layerd = Dense(64, activation="relu")(word_lstm_outputsd)
        regression_output = Dense(1, activation="sigmoid", name="regression_output")(merged_layerd)
        classification_output = Dense(num_class, activation="softmax", name="classification_output")(merged_layerd)
        prediction_model = Model(inputs=word_inputs,
                                 outputs=[regression_output, classification_output])

    prediction_model.compile(loss={"regression_output": "mean_squared_error",
                                   "classification_output": "categorical_crossentropy"}, optimizer='adam',
                             metrics={"regression_output": metrics.mae,
                                   "classification_output": "accuracy"})
    return prediction_model


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


def pearson_correlation(y_true, y_pred):
    l1 = list(y_true)
    l2 = [p[0] for p in y_pred]
    return np.corrcoef(l1, l2)[0, 1]


def acc(y_true, y_pred):
    l1 = [np.argmax(p) for p in list(y_true)]
    l2 = [np.argmax(p) for p in y_pred]
    return np.sum([1 for x, y in zip(l1, l2) if x == y]) / (len(l1) * 1.0)


def encode(arr, word2id, max_len=word_lstm_sentiment_model.MAX_SEQUENCE_LENGTH):
    res = []
    for x in arr:
        res.append(data_utils.encode_sentence(x, word2id, None, threshold=0, add_unknowns=False))
    x = sequence.pad_sequences(res, maxlen=max_len)
    return x


def train_model(name, regression_train_data_path, classification_train_data_path,
                regression_dev_data_path, classification_dev_path,
                word_index, use_sentiment_features=True, use_lexicon_features=True,
                feature_extraction_model=None, batch_size=32, epsilon=0.03, embedding_matrix=None):
    max_pearson = 0.0
    max_accuracy = 0.0
    early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    lexicon_feature_extractor = LexiconFeatureExtractor()
    print "Preparing Data for {} model ...".format(name)
    train_x, train_y_labels = data_utils.load_affect_data(classification_train_data_path, is_label_numeric=False)
    train_x2, train_y_scores = data_utils.load_affect_data(regression_train_data_path)
    assert len(train_x) == len(train_x2)
    for x1, x2 in zip (train_x, train_x2):
        if x1 != x2:
            raise Exception("Instances in regression data and classification data are different!")
    train_y_labels = to_categorical(train_y_labels)
    train_y_scores = np.array(train_y_scores)
    dev_x, dev_y_labels = data_utils.load_affect_data(classification_dev_path, is_label_numeric=False)
    _, dev_y_scores = data_utils.load_affect_data(regression_dev_data_path)
    dev_y_scores = np.array(dev_y_scores)
    dev_y_labels = to_categorical(dev_y_labels)
    encoded_train_x = encode(train_x, word_index)
    encoded_dev_x = encode(dev_x, word_index)
    if use_sentiment_features and use_lexicon_features:
        sentiment_features_train = feature_extraction_model.predict(encoded_train_x)
        lexicon_features_train = [extract_lexicon_features(instance, lexicon_feature_extractor) for instance in train_x]
        lexicon_features_train = np.array(lexicon_features_train)
        sentiment_features_dev = feature_extraction_model.predict(encoded_dev_x)
        lexicon_features_dev = [extract_lexicon_features(instance, lexicon_feature_extractor) for instance in dev_x]
        lexicon_features_dev = np.array(lexicon_features_dev)
        model = create_prediction_model(len(sentiment_features_train[0]), len(lexicon_features_train[0]),
                                        len(train_y_labels[0]), len(word_index), embedding_matrix=embedding_matrix)
        print model.summary()
        print "Start training for {} model ...".format(name)
        max_harmonic_mean = 0.0
        while True:
            model.fit(x=[encoded_train_x, sentiment_features_train, lexicon_features_train],
                      y={"regression_output": train_y_scores, "classification_output": train_y_labels},
                      validation_data=(
                          [encoded_dev_x, sentiment_features_dev, lexicon_features_dev],
                          {"regression_output": dev_y_scores, "classification_output": dev_y_labels}
                      ),
                      epochs=1, callbacks=[early_stop], batch_size=batch_size, shuffle=True)
            predictions = model.predict([encoded_dev_x, sentiment_features_dev, lexicon_features_dev])
            pearson = pearson_correlation(dev_y_scores, predictions[0])
            accuracy = acc(dev_y_labels, predictions[1])
            harmonic_mean = 2 * accuracy * pearson / (accuracy + pearson)
            if harmonic_mean > max_harmonic_mean:
                max_harmonic_mean = harmonic_mean
                max_pearson = pearson
                max_accuracy = accuracy
                print "Max harmonic mean of pearson and accuracy is increased: {} ...".format(max_harmonic_mean)
                print "Saving {} model ...".format(name)
                model.save("resources/saved_models/{}.model".format(name))
            print "pearson correlation on train set={}".format(pearson)
            print "classification accuracy on train set={}".format(accuracy)
            if harmonic_mean < max_harmonic_mean - epsilon:
                break
    elif use_lexicon_features and not use_sentiment_features:
        lexicon_features_train = [extract_lexicon_features(instance, lexicon_feature_extractor) for instance in train_x]
        lexicon_features_train = np.array(lexicon_features_train)
        lexicon_features_dev = [extract_lexicon_features(instance, lexicon_feature_extractor) for instance in dev_x]
        lexicon_features_dev = np.array(lexicon_features_dev)
        model = create_prediction_model(0, len(lexicon_features_train[0]),
                                        len(train_y_labels[0]), len(word_index), embedding_matrix=embedding_matrix)
        print model.summary()
        print "Start training for {} model ...".format(name)
        max_harmonic_mean = 0.0
        while True:
            model.fit(x=[encoded_train_x, lexicon_features_train],
                      y={"regression_output": train_y_scores, "classification_output": train_y_labels},
                      validation_data=(
                          [encoded_dev_x, lexicon_features_dev],
                          {"regression_output": dev_y_scores, "classification_output": dev_y_labels}
                      ),
                      epochs=1, callbacks=[early_stop], batch_size=batch_size, shuffle=True)
            predictions = model.predict([encoded_dev_x, lexicon_features_dev])
            pearson = pearson_correlation(dev_y_scores, predictions[0])
            accuracy = acc(dev_y_labels, predictions[1])
            harmonic_mean = 2 * accuracy * pearson / (accuracy + pearson)
            if harmonic_mean > max_harmonic_mean:
                max_harmonic_mean = harmonic_mean
                max_pearson = pearson
                max_accuracy = accuracy
                print "Max harmonic mean of pearson and accuracy is increased: {} ...".format(max_harmonic_mean)
                print "Saving {} model ...".format(name)
                model.save("resources/saved_models/{}.model".format(name))
            print "pearson correlation on train set={}".format(pearson)
            print "classification accuracy on train set={}".format(accuracy)
            if harmonic_mean < max_harmonic_mean - epsilon:
                break
    elif not use_lexicon_features and use_sentiment_features:
        sentiment_features_train = feature_extraction_model.predict(encoded_train_x)
        sentiment_features_dev = feature_extraction_model.predict(encoded_dev_x)
        model = create_prediction_model(len(sentiment_features_train[0]), 0,
                                        len(train_y_labels[0]), len(word_index), embedding_matrix=embedding_matrix)
        print model.summary()
        print "Start training for {} model ...".format(name)
        max_harmonic_mean = 0.0
        while True:
            model.fit(x=[encoded_train_x, sentiment_features_train],
                      y={"regression_output": train_y_scores, "classification_output": train_y_labels},
                      validation_data=(
                          [encoded_dev_x, sentiment_features_dev],
                          {"regression_output": dev_y_scores, "classification_output": dev_y_labels}
                      ),
                      epochs=1, callbacks=[early_stop], batch_size=batch_size, shuffle=True)
            predictions = model.predict([encoded_dev_x, sentiment_features_dev])
            pearson = pearson_correlation(dev_y_scores, predictions[0])
            accuracy = acc(dev_y_labels, predictions[1])
            harmonic_mean = 2 * accuracy * pearson / (accuracy + pearson)
            if harmonic_mean > max_harmonic_mean:
                max_harmonic_mean = harmonic_mean
                max_pearson = pearson
                max_accuracy = accuracy
                print "Max harmonic mean of pearson and accuracy is increased: {} ...".format(max_harmonic_mean)
                print "Saving {} model ...".format(name)
                model.save("resources/saved_models/{}.model".format(name))
            print "pearson correlation on train set={}".format(pearson)
            print "classification accuracy on train set={}".format(accuracy)
            if harmonic_mean < max_harmonic_mean - epsilon:
                break
    else:
        model = create_prediction_model(0, 0,
                                        len(train_y_labels[0]), len(word_index), embedding_matrix=embedding_matrix)
        print model.summary()
        print "Start training for {} model ...".format(name)
        max_harmonic_mean = 0.0
        while True:
            model.fit(x=encoded_train_x,
                      y={"regression_output": train_y_scores, "classification_output": train_y_labels},
                      validation_data=(
                          encoded_dev_x,
                          {"regression_output": dev_y_scores, "classification_output": dev_y_labels}
                      ),
                      epochs=1, callbacks=[early_stop], batch_size=batch_size, shuffle=True)
            predictions = model.predict(encoded_dev_x)
            pearson = pearson_correlation(dev_y_scores, predictions[0])
            accuracy = acc(dev_y_labels, predictions[1])
            harmonic_mean = 2 * accuracy * pearson / (accuracy + pearson)
            if harmonic_mean > max_harmonic_mean:
                max_harmonic_mean = harmonic_mean
                max_pearson = pearson
                max_accuracy = accuracy
                print "Max harmonic mean of pearson and accuracy is increased: {} ...".format(max_harmonic_mean)
                print "Saving {} model ...".format(name)
                model.save("resources/saved_models/{}.model".format(name))
            print "pearson correlation on train set={}".format(pearson)
            print "classification accuracy on train set={}".format(accuracy)
            if harmonic_mean < max_harmonic_mean - epsilon:
                break
    return max_pearson, max_accuracy


def load_models(model_names):
    feature_extraction_model, word_index = create_feature_extraction_model()
    prdiction_models = [load_model("resources/saved_models/{}_en_sentiment_lexicon.model".format(model_name)) for model_name in model_names]
    return feature_extraction_model, word_index, prdiction_models


def train_models(model_names):
    with open("results.txt", "w") as f:
        print "Creating Feature Extraction model"
        feature_extraction_model, word_index = create_feature_extraction_model()
        for model_name in model_names:
            pearson, accuracy = train_model("{}_en_sentiment_lexicon".format(model_name), "data/semeval2018 task 1/EI-reg-En-{}-train.txt".format(model_name),
                                        "data/semeval2018 task 1/EI-oc-En-{}-train.txt".format(model_name),
                                        "data/semeval2018 task 1/2018-EI-reg-En-{}-dev.txt".format(model_name),
                                        "data/semeval2018 task 1/2018-EI-oc-En-{}-dev.txt".format(model_name),
                                        word_index, use_sentiment_features=True, use_lexicon_features=True,
                                        feature_extraction_model=feature_extraction_model)
            f.write("Model:{}_en_sentiment_lexicon\tPearson:{}\tAccuracy:{}\n".format(model_name, pearson, accuracy))
            f.flush()
            pearson, accuracy = train_model("{}_en_sentiment".format(model_name), "data/semeval2018 task 1/EI-reg-En-{}-train.txt".format(model_name),
                                        "data/semeval2018 task 1/EI-oc-En-{}-train.txt".format(model_name),
                                        "data/semeval2018 task 1/2018-EI-reg-En-{}-dev.txt".format(model_name),
                                        "data/semeval2018 task 1/2018-EI-oc-En-{}-dev.txt".format(model_name),
                                        word_index, use_sentiment_features=True, use_lexicon_features=False,
                                        feature_extraction_model=feature_extraction_model)
            f.write("Model:{}_en_sentiment\tPearson:{}\tAccuracy:{}\n".format(model_name, pearson, accuracy))
            f.flush()
            pearson, accuracy = train_model("{}_en_lexicon".format(model_name), "data/semeval2018 task 1/EI-reg-En-{}-train.txt".format(model_name),
                                        "data/semeval2018 task 1/EI-oc-En-{}-train.txt".format(model_name),
                                        "data/semeval2018 task 1/2018-EI-reg-En-{}-dev.txt".format(model_name),
                                        "data/semeval2018 task 1/2018-EI-oc-En-{}-dev.txt".format(model_name),
                                        word_index, use_sentiment_features=False, use_lexicon_features=True,
                                        feature_extraction_model=feature_extraction_model)
            f.write("Model:{}_en_lexicon\tPearson:{}\tAccuracy:{}\n".format(model_name, pearson, accuracy))
            f.flush()
            pearson, accuracy = train_model("{}_en".format(model_name), "data/semeval2018 task 1/EI-reg-En-{}-train.txt".format(model_name),
                                        "data/semeval2018 task 1/EI-oc-En-{}-train.txt".format(model_name),
                                        "data/semeval2018 task 1/2018-EI-reg-En-{}-dev.txt".format(model_name),
                                        "data/semeval2018 task 1/2018-EI-oc-En-{}-dev.txt".format(model_name),
                                        word_index, use_sentiment_features=False, use_lexicon_features=False,
                                        feature_extraction_model=feature_extraction_model)
            f.write("Model:{}_en\tPearson:{}\tAccuracy:{}\n\n".format(model_name, pearson, accuracy))
            f.flush()

if __name__ == "__main__":
    model_names = ["joy"]
    train_models(model_names)
