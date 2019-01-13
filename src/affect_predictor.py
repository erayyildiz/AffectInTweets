# -*- coding: utf-8 -*-
import numpy as np
import sys
from os import listdir
from os.path import isfile, join
from affect_model_trainer import encode, create_prediction_model, create_feature_extraction_model
from lexicon_features import LexiconFeatureExtractor
from keras.models import load_model


class AffectPredictor(object):
    SENTIMENT_FEATURES_LEN = 400
    LEXICON_FEATURES_LEN = 18
    CLASS_DIC = {0: "0: no __ can be inferred", 1: "1: low amount of __ can be inferred", 2: "2: moderate amount of __ can be inferred", 3: "3: high amount of __ can be inferred"}
    EMOTION_CATEGORIES = ["anger", "fear", "joy", "sadness", "valence"]
    EMOTION_NAMES = ["öfke", "korku", "neşe", "üzüntü", "valans"]

    def __init__(self):
        print "Initializing lexicon feature extractor..."
        self.lexicon_feature_extractor = LexiconFeatureExtractor()
        print "Loading sentiment pretrained model..."
        self.feature_extraction_model, self.word2id = create_feature_extraction_model()
        self.prediction_models = []
        print "Loading emotion models..."
        for emotion_category in AffectPredictor.EMOTION_CATEGORIES:
			print "Loading {} model...".format(emotion_category)
            self.prediction_models.append(load_model("resources/saved_models/{}_en_sentiment_lexicon.model"
                                                     .format(emotion_category)))
        print "Done."

    def predict(self, tweet):
        res = {}
        encoded_tweet = encode([tweet], self.word2id)
        sentiment_features = self.feature_extraction_model.predict(encoded_tweet)
        lexicon_features = [self.lexicon_feature_extractor.extract_feature(tweet)]
        lexicon_features = np.array(lexicon_features)
        for emotion_category, model in zip(AffectPredictor.EMOTION_NAMES, self.prediction_models):
            model_predictions = model.predict([encoded_tweet, sentiment_features, lexicon_features])
            reg_result = model_predictions[0][0][0]
            reg_result = int(round(reg_result * 100))
            classification_result = AffectPredictor.CLASS_DIC[np.argmax(model_predictions[1][0])]
            res[emotion_category] = (reg_result, classification_result)
        return res
		
	def predict_one(self, tweet, emotion_category, task):
		model = self.prediction_models[AffectPredictor.EMOTION_CATEGORIES.index(emotion_category)]
		encoded_tweet = encode([tweet], self.word2id)
        sentiment_features = self.feature_extraction_model.predict(encoded_tweet)
        lexicon_features = [self.lexicon_feature_extractor.extract_feature(tweet)]
        lexicon_features = np.array(lexicon_features)
		model_predictions = model.predict([encoded_tweet, sentiment_features, lexicon_features])
		if task == "reg":
			return model_predictions[0][0][0]
		elif task == "oc":
			return AffectPredictor.CLASS_DIC[np.argmax(model_predictions[1][0])].replace("__", emotion_category)
		
def main(argv):
    affect_predictor = AffectPredictor()
	testfiles = [f for f in listdir("data/test sets/") if isfile(join("data/test sets/", f))]
	for test_file in testfiles:
		if "-reg-" in test_file:
			task = "reg"
		elif "-oc-" in test_file:
			task =	"oc"
		else:
			raise Exception("unknown task")
		with open("submit/"+test_file, "w") as w:
			with open("data/test sets/"+test_file, "r") as f:
				for i, line in enumerate(f):
					if i == 0:
						continue
					splits = line.split("\t")
					id = splits[0].strip()
					tweet = splits[1].strip()
					emotion_category = splits[2].strip()
					res = affect_predictor.predict_one(tweet, emotion_category, task)
					w.write("{}\t{}\t{}\t{}\n".format(id, tweet, emotion_category, res))

if __name__ == "__main__":
    main(sys.argv[1:])
