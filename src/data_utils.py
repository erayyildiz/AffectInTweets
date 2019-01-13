import pandas
import sys
from keras.utils.np_utils import to_categorical
import word2vecReaderUtils
from nltk.tokenize import TweetTokenizer


def ddict2dict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = ddict2dict(v)
    return dict(d)


def load_sentiment_data(file_path, frac=1, word2id={}, add_unknowns=True):
    xs = []
    ys = []
    wordcounts = {}
    column_names = ['y', '', '', '', '', 'x']
    df = pandas.read_csv(file_path, header=None, names=column_names)
    df = df.sample(frac=frac)
    instance_count = len(df)
    df = df[['x', 'y']]
    for i, row in enumerate(df.iterrows()):
        sys.stdout.write("\r%d / %d      " % (i, instance_count))
        tokens = word2vecReaderUtils.tokenize(row[1]['x'])
        for token in tokens:
            if token in wordcounts:
                wordcounts[token] += 1
            else:
                wordcounts[token] = 0

    for i, row in enumerate(df.iterrows()):
        sys.stdout.write("\r%d / %d      " % (i, instance_count))
        encoded = encode_sentence(row[1]['x'], word2id, wordcounts, threshold=5, add_unknowns=add_unknowns)
        if encoded:
            xs.append(encoded)
            cur_y = int(row[1]['y'])
            if cur_y != 0:
                cur_y = 1
            ys.append(cur_y)
        if i % 1000 == 0:
            sys.stdout.flush()
    return xs, to_categorical(ys), word2id


def load_affect_data(file_path, is_label_numeric=True, label_index=3, text_index=1):
    x = []
    y = []
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                # Headers
                continue
            columns = line.split("\t")
            if len(columns) > label_index and columns[label_index]:
                if is_label_numeric:
                    try:
                        val = float(columns[label_index])
                        y.append(val)
                        x.append(columns[text_index].strip("\r\n\t "))
                    except ValueError:
                        print columns[label_index]
                        continue
                else:
                    label = columns[label_index].split(":")[0]
                    val = int(label)
                    y.append(val)
                    x.append(columns[text_index].strip("\r\n\t "))
    return x, y


def encode_sentence(sentence, word2id, wordcounts, threshold=5, add_unknowns=True):
    res = []
    try:
        tokens = TweetTokenizer().tokenize(sentence)
        for token in tokens:
            if threshold == 0 or wordcounts[token] >= threshold:
                if token not in word2id:
                    if add_unknowns:
                        word2id[token] = len(word2id) + 1
                        res.append(word2id[token])
                    else:
                        new_token = token
                        if new_token.startswith("#") and len(new_token) > 1:
                            new_token = new_token[1:]
                        if "'" in new_token:
                            new_token = new_token[:new_token.rfind("'")]
                        if new_token in word2id:
                            res.append(word2id[new_token])
                        else:
                            new_token = new_token.lower()
                            if new_token in word2id:
                                res.append(word2id[new_token])
                            elif len(new_token) > 1:
                                res.append(0)
                else:
                    res.append(word2id[token])
                    if token.startswith("#") and len(token) > 1 and token[1:] in word2id:
                        res.append(word2id[token[1:]])
    except UnicodeDecodeError:
        return None
    if len(res) > 0:
        return res
    else:
        return [0]

