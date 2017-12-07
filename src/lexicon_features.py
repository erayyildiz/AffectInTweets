from nltk.tokenize import TweetTokenizer

class LexiconFeatureExtractor:

    def __init__(self, afinn_lexicon_file_path="resources/lexicons/AFINN-en-165.txt",
                 afinn_emoticon_lexicon_file_path="resources/lexicons/AFINN-emoticon-8.txt",
                 bing_liu_lexicon_file_path="resources/lexicons/BingLiu.txt",
                 mpqa_lexicon_file_path="resources/lexicons/mpqa.txt"):
        print "Loading AFINN lexicons..."
        self.afinn_lexicon = LexiconFeatureExtractor._read_standart_lexicon(afinn_lexicon_file_path)
        self.afinn_emoticon_lexicon = LexiconFeatureExtractor._read_standart_lexicon(afinn_emoticon_lexicon_file_path)
        print "Loading BingLiu lexicon..."
        self.bingliu_lexicon = LexiconFeatureExtractor._read_standart_lexicon(bing_liu_lexicon_file_path)
        print "Loading MPQA lexicon..."
        self.mpqa_lexicon = LexiconFeatureExtractor._read_standart_lexicon(mpqa_lexicon_file_path)

    def extract_feature(self, input_txt):
        res = []
        # AFINN Lexicons
        res.append(LexiconFeatureExtractor.calculate_score_word_based(self.afinn_lexicon, input_txt))
        res.append(LexiconFeatureExtractor.calculate_score_word_based(self.afinn_emoticon_lexicon, input_txt))
        # BING LIU Lexicon
        res.append(LexiconFeatureExtractor.calculate_score_word_based(self.bingliu_lexicon, input_txt))
        # MPQA Lexicon
        res.append(LexiconFeatureExtractor.calculate_score_word_based(self.mpqa_lexicon, input_txt))
        return res

    @staticmethod
    def _read_standart_lexicon(file_path, delimeter="\t"):
        res = {}
        with(open(file_path, "r")) as f:
            for line in f:
                columns = line.strip().split(delimeter)
                if len(columns) > 1:
                    res[columns[0]] = float(columns[-1])
        return res

    @staticmethod
    def calculate_score_word_based(lexicon, input_txt):
        score = 0.0
        input_words = input_txt.split()
        for k, v in lexicon.iteritems():
            if k in input_words:
                score += v
        return score








