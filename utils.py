import os
import re
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9'\s]", "", text)
    return text

def data_generator(file_path, batch_size, max_features, max_len, tokenizer=None):
    with open(file_path, "r") as f:
        texts, labels = [], []
        while True:
            line = f.readline()
            if not line:
                f.seek(0)
                continue

            label, text = line.split(" ", 1)
            label = 1 if label == '__label__2' else 0

            text = clean_text(text)
            texts.append(text)
            labels.append(label)

            if len(texts) == batch_size:
                if tokenizer is None:
                    tokenizer = Tokenizer(num_words=max_features, oov_token='<OOV>')
                    tokenizer.fit_on_texts(texts)

                sequences = tokenizer.texts_to_sequences(texts)
                padded_sequences = pad_sequences(sequences, maxlen=max_len, truncating='post', padding='post')

                yield (np.array(padded_sequences), np.array(labels))

                texts, labels = [], []

def get_tokenizer(file_path, max_features):
    with open(file_path, "r") as f:
        texts = [clean_text(line.split(" ", 1)[1]) for line in f.readlines()]
    
    tokenizer = Tokenizer(num_words=max_features, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    return tokenizer
