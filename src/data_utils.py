import pandas
import sys
from keras.utils.np_utils import to_categorical
import word2vecReaderUtils



def ddict2dict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = ddict2dict(v)
    return dict(d)


def load_sentiment_data(file_path, frac=1, word2id={}, add_unknowns=True):
    xs = []
    ys = []
    column_names = ['y', '', '', '', '', 'x']
    df = pandas.read_csv(file_path, header=None, names=column_names)
    df = df.sample(frac=frac)
    instance_count = len(df)
    df = df[['x', 'y']]
    for i, row in enumerate(df.iterrows()):
        sys.stdout.write("\r%d / %d      " % (i, instance_count))
        encoded = encode_sentence(row[1]['x'], word2id, add_unknowns=add_unknowns)
        if encoded:
            xs.append(encoded)
            cur_y = int(row[1]['y'])
            if cur_y != 0:
                cur_y = 1
            ys.append(cur_y)
        if i % 1000 == 0:
            sys.stdout.flush()
    return xs, to_categorical(ys), word2id


def encode_sentence(sentence, word2id, add_unknowns=True):
    res = []
    try:
        tokens = word2vecReaderUtils.tokenize(sentence)
        for token in tokens:
            if token not in word2id:
                if add_unknowns:
                    word2id[token] = len(word2id) + 1
                    res.append(word2id[token])
                else:
                    res.append(0)
            else:
                res.append(word2id[token])
    except UnicodeDecodeError:
        return None
    return res

