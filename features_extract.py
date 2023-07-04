import pandas as pd
import string
import re

from transformers import BertTokenizerFast
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import defaultdict

from stop_words import STOP_WORDS
# PUNKT does not contain # (for bert) and ` (for words like `ה)
PUNCT = ["\t", "\"", "`", "´", "״", "–", "“", "”", "$", "%", "&", "\\", "\'", "(", ")", "*", "+", ",", "-", ".", "/", ":", ";", "<", "=", ">", "?", "@", "[", "]", "^", "_", "{", "|", "}", "~"]
INVALID_CHARS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "q", "w", "e", "r", "t", "y", "u", "i", "o", "p", "a", "d", "f", "g", "h", "j", "k", "l", "z", "x", "c", "v", "b", "n", "m", "Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "A", "S", "D", "F", "G", "H", "J", "K", "L", "Z", "X", "C", "V", "B", "N", "M"]


##### Tokenization Functions #####
def my_own_tokenizer(text):
    translator = str.maketrans(string.punctuation+"\n"+"\t", ' '*(len(string.punctuation) + 2))
    text = text.translate(translator)

    re.sub(' +',' ', text).strip()
    tokens = text.split(' ')

    return tokens

alephbert_tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')
# alephbertgimmel_tokenizer = AutoTokenizer.from_pretrained('dicta-il/alephbertgimmel-base')

def bert_tokenizer(text):
    global alephbert_tokenizer

    tokens = alephbert_tokenizer.tokenize(text)
    return tokens

def clean_tokens(tokens, punct = True, stop_words = False):
    tokens = [tok for tok in tokens if len(tok) > 1]
    
    if punct:
        tokens = [tok for tok in tokens if (
            [c for c in tok if ((c in PUNCT) or (c in INVALID_CHARS))] == []
            )
        ]

    if stop_words:
        tokens = [tok for tok in tokens if tok not in STOP_WORDS]
    
    return tokens

def create_tokens_columns(df, sw = True):
    df["Q_tokens_no_sw"] = df.apply(lambda r: my_own_tokenizer(r['Question']), axis=1).apply(
        lambda x: clean_tokens(x, stop_words=sw)
    )

def create_bert_tokens_columns(df, sw = True):
    col_name = "Q_bert_tokens{}".format("_no_sw" if sw else "")
    df[col_name] = df.apply(lambda r: bert_tokenizer(r['Question']), axis=1).apply(
        lambda x: clean_tokens(x, stop_words=sw)
    )


##### Vectorization Functions #####
def create_frequency_dict(texts):
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    return frequency

def filter_tokens_by_freq(texts, min_freq = 10):
    freq_dict = create_frequency_dict(texts)
    docs = [
        [token for token in text if freq_dict[token] >= min_freq]
        for text in texts
    ]
    return docs

def create_filter_vocab(tokens_corpus, min_freq):
    filtered_tokens_corpus = filter_tokens_by_freq(tokens_corpus, min_freq)
    filtered_vocab = {item for sublist in filtered_tokens_corpus for item in sublist}
    return filtered_vocab

def dummy(doc):
    return doc

def create_count_vector_df(tokens_corpus, vocab):
    cv = CountVectorizer(
        tokenizer=dummy,
        preprocessor=dummy,
        vocabulary=vocab
    )
    count_vector = cv.fit_transform(tokens_corpus)
    count_vect_df = pd.DataFrame(count_vector.todense(), columns=cv.get_feature_names_out())
    return count_vect_df

def create_tfidf_vector_df(tokens_corpus, vocab):
    tfidf = TfidfVectorizer(
        tokenizer=dummy,
        preprocessor=dummy,
        vocabulary=vocab
    )
    tfidf_vector = tfidf.fit_transform(tokens_corpus)
    tfidf_vect_df = pd.DataFrame(tfidf_vector.todense(), columns=tfidf.get_feature_names_out())
    return tfidf_vect_df


##### Features Extraction #####

def concat_feat_with_data(data, feat_df):
    col_from_data = ["Question", "Answer", "QuestionLen", "AnswerLen", "Gender"]
    data_with_feat = pd.concat([data[col_from_data], feat_df], axis=1)
    return data_with_feat

def write_data_with_feat_to_csv(df, folder, file_name):
    df.to_csv(folder + file_name)