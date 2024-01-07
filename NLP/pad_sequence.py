import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

learn_text = [
    'Weather is great today',
    'Weather was great yesterday',
    'Weather is great tomorrow',
    'cold weather',
    'warm weather happens in June'
    ]
test_text = [
    'weather is great in London',
    'how is the weather today',
    'yearly weather average',
    'weather news',
    'is this hotter weather than usual'
    ]

tokenizer = Tokenizer(num_words = 100, oov_token="<unknown>")
tokenizer.fit_on_texts(learn_text)
word_index = tokenizer.word_index

print('\nLearning text:')
print(learn_text)

print('\nWord Index:')
print(word_index)

raw_sequences = tokenizer.texts_to_sequences(learn_text)

print('\n Raw Sequences data (learning text):')
print(raw_sequences)

# if we don't supply default the longest sequence length is taken as maxlen.
# if we don't supply 'post' then padding and tructating both are from the
# beginning of the sequence.
padded = pad_sequences(raw_sequences,
                       padding='post',  # Add zero to end in short sequence
                       truncating='post', # longer ones truncate from the end
                       maxlen=10) # max len 10

print('\nPadded Sequences with maxlen=10:')
print(padded)

test_seq = tokenizer.texts_to_sequences(test_text)

print('\nTest data:')
print(test_text)

print('\nSequence data (test text):')
print('{}\n'.format(test_seq))
