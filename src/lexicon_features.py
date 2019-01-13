from nltk.tokenize import TweetTokenizer


class LexiconFeatureExtractor:
    def __init__(self, afinn_lexicon_file_path="resources/lexicons/AFINN-en-165.txt",
                 afinn_emoticon_lexicon_file_path="resources/lexicons/AFINN-emoticon-8.txt",
                 bing_liu_lexicon_file_path="resources/lexicons/BingLiu.txt",
                 mpqa_lexicon_file_path="resources/lexicons/mpqa.txt"):
        print("Loading AFINN lexicons...")
        self.afinn_lexicon = LexiconFeatureExtractor._read_standart_lexicon(afinn_lexicon_file_path)
        self.afinn_emoticon_lexicon = LexiconFeatureExtractor._read_standart_lexicon(afinn_emoticon_lexicon_file_path)
        print("Loading BingLiu lexicon...")
        self.bingliu_lexicon = LexiconFeatureExtractor._read_standart_lexicon(bing_liu_lexicon_file_path)
        print("Loading MPQA lexicon...")
        self.mpqa_lexicon = LexiconFeatureExtractor._read_standart_lexicon(mpqa_lexicon_file_path)
        print("Loading NRC - Hashtag - Emotion - Lexicon")
        self.nrc_hash_emo_lexicon = LexiconFeatureExtractor \
            ._read_labeled_lexicon("resources/lexicons/NRC-Hashtag-Emotion-Lexicon-v0.2.txt")
        print("Loading NRC - AffectIntensity - Lexicon")
        self.nrc_affect_intensity_lexicon = LexiconFeatureExtractor \
            ._read_labeled_lexicon("resources/lexicons/NRC-AffectIntensity-Lexicon.txt")
        print("Loading SentiStrength EmoticonLookupTable")
        self.emoticon_lookup_lexicon = LexiconFeatureExtractor \
            ._read_standart_lexicon("resources/lexicons/EmoticonLookupTable.txt")
        print("Loading SentiStrength EmotionLookupTable")
        self.emotion_lookup_lexicon = LexiconFeatureExtractor \
            ._read_standart_lexicon("resources/lexicons/EmotionLookupTable.txt")

    def extract_feature(self, input_txt):
        res = [LexiconFeatureExtractor.calculate_score_word_based(self.afinn_lexicon, input_txt),
               LexiconFeatureExtractor.calculate_score_word_based(self.afinn_emoticon_lexicon, input_txt),
               LexiconFeatureExtractor.calculate_score_word_based(self.bingliu_lexicon, input_txt),
               LexiconFeatureExtractor.calculate_score_word_based(self.mpqa_lexicon, input_txt)]

        # NRC - Hashtag - Emotion - Lexicon
        res += LexiconFeatureExtractor.calculate_score_labeled(self.nrc_hash_emo_lexicon, input_txt)
        # NRC - Affect intensity - Lexicon
        res += LexiconFeatureExtractor.calculate_multiscore(self.nrc_affect_intensity_lexicon, input_txt)
        # SentiStrength - Emoticon - Lexicon
        res.append(LexiconFeatureExtractor.calculate_score_word_based(self.emoticon_lookup_lexicon, input_txt))
        # SentiStrength - Emotion - Lexicon
        res.append(LexiconFeatureExtractor.calculate_score_word_based(self.emotion_lookup_lexicon, input_txt))

        return res

    @staticmethod
    def _read_standart_lexicon(file_path, delimeter="\t"):
        res = {}
        with(open(file_path, "r")) as f:
            for line in f:
                columns = line.strip().split(delimeter)
                if len(columns) > 1:
                    res[" ".join(columns[:-1]).strip(" ")] = float(columns[-1])
        return res

    @staticmethod
    def _read_multi_score_lexicon(file_path, delimeter="\t", ):
        res = {}
        with(open(file_path, "r")) as f:
            for line in f:
                scores = []
                columns = line.strip().split(delimeter)
                for i in range(1, len(columns)):
                    scores.append(float(columns[i]))
                res[columns[0]] = scores
        return res

    @staticmethod
    def _read_labeled_lexicon(file_path, delimeter="\t",
                              label_index=0, feature_index=1, score_index=2):
        res = {}
        with(open(file_path, "r")) as f:
            for line in f:
                columns = line.strip().split(delimeter)
                if len(columns) > 2:
                    if columns[label_index] not in res:
                        res[columns[label_index]] = {}
                    res[columns[label_index]][columns[feature_index]] = float(columns[score_index])
        return res

    @staticmethod
    def calculate_score_word_based(lexicon, input_txt):
        score = 0.0
        input_words = [t.encode("utf-8") for t in TweetTokenizer().tokenize(input_txt)]
        for k, v in lexicon.items():
            if " " not in k and k in input_words:
                score += v
            elif " " in k and LexiconFeatureExtractor.contains_all(k, input_words):
                score += v
        return score

    @staticmethod
    def calculate_multiscore(lexicon, input_txt, score_count=4):
        res = [0.0 for _ in range(score_count)]
        input_words = [t.encode("utf-8") for t in TweetTokenizer().tokenize(input_txt)]
        for label, d in lexicon.items():
            for k, v in d.items():
                scores = []
                if " " not in k and k in input_words:
                    scores.append(v)
                elif " " in k and LexiconFeatureExtractor.contains_all(k, input_words):
                    scores.append(v)
            for i in range(len(scores)):
                res[i] += scores[i]
        return res

    @staticmethod
    def calculate_score_labeled(lexicon, input_txt):
        res = []
        score = 0.0
        input_words = [t.encode("utf-8") for t in TweetTokenizer().tokenize(input_txt)]
        for label, d in lexicon.items():
            for k, v in d.items():
                score = 0.0
                if " " not in k and k in input_words:
                    score += v
                elif " " in k and LexiconFeatureExtractor.contains_all(k, input_words):
                    score += v
            res.append(score)
        return res

    @staticmethod
    def contains_all(words1, words2):
        for w in words1.split():
            if w not in words2:
                return False
        return True
