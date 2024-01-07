import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

learn_text = [
    'Weather is great today',
    'Weather was great yesterday',
    'Weather is great tomorrow'
    ]
test_text = [
    'weather is great in London',
    'how is the weather today'
    ]

tokenizer = Tokenizer(num_words = 100)
tokenizer.fit_on_texts(learn_text)
word_index = tokenizer.word_index

print('\nLearning text:')
print(learn_text)

print('\nWord Index:')
print(word_index)

sequence_data = tokenizer.texts_to_sequences(learn_text)

print('\nSequence data (learning text):')
print(sequence_data)

test_seq = tokenizer.texts_to_sequences(test_text)

print('\nTest data:')
print(test_text)

print('\nSequence data (test text):')
print('{}\n'.format(test_seq))
