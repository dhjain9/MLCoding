import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

learn_text = [
    'Weather is great today',
    'Weather was great yesterday',
    'Weather is great tomorrow'
    ]

tokenizer = Tokenizer(num_words = 100)
tokenizer.fit_on_texts(learn_text)
word_index = tokenizer.word_index

print(word_index)
