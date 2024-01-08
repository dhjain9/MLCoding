import csv
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def read_common_words(filename):
  with open(filename, 'r') as wfile:
    reader = csv.reader(wfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    data = []
    for row in reader:
      # Handle multiple fields entries within a row if necessary e.g. a,b,c,
      new_row = [field for field in row if field not in ['']]
      data.extend(new_row)
      
    print('common words: total: {}'.format(len(data)))
    print(data)
    print('\n')
  return data

def ignore_common_words(sentence, common_words):
  # Sentence converted to lowercase-only
  sentence = sentence.lower() 
  words = sentence.split()
  clean_words = [word for word in words if word.lower() not in common_words]
  sentence = ' '.join(clean_words)
  return sentence


def get_sentences_and_labels(filename):
  sentences = []
  labels = []
  with open(filename, 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    ignored_first_row = next(reader)
    for label, sentence in reader:
      labels.append(label)
      sentences.append(sentence)
  return sentences, labels


common_words = read_common_words("./data/common_words.csv")
sentences, labels = get_sentences_and_labels("./data/bbc-text-minimal.csv")
sentences = [ignore_common_words(sentence, common_words) for sentence in sentences]


print('Total sentences in the dataset {}'.format(len(sentences)))
print('Num labels in the dataset'.format(len(labels)))
print(labels)
print('\n')

# tokenizer for sentences      
tokenizer = Tokenizer(num_words = 1000000, oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

print('Tokenizer for sentences:')
print('number of words {}'.format(len(word_index)))
print('word_index: len = {}'.format(len(word_index)))
print('\n')
      
sequences = tokenizer.texts_to_sequences(sentences)
padded_sequences = pad_sequences(sequences, padding='post')    

# tokenizer for labels
label_tokenizer = Tokenizer(num_words=100000)
label_tokenizer.fit_on_texts(labels)
label_word_index = label_tokenizer.word_index

print('Tokenizer for labels:')
print('number of words {}'.format(len(label_word_index)))
print('word_index:')
print(label_word_index)
print('\n')

label_sequences = label_tokenizer.texts_to_sequences(labels)

print("Padded Sequences for sentences:")
print(padded_sequences)
print("\n")
print("Sequences for labels:")
print(label_sequences)
print("\n")

