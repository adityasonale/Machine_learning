import numpy as np
from keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import re
import nltk 
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pickle
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


model = load_model("D:\\vs code\python\DeepLearning\models\spamDetector.h5")

stemmer = PorterStemmer()

# text preprocessing

def preprocessing(email):
    text = []
    sentence = re.sub('[^a-zA-z]',' ',email)
    sentence = sentence.lower()
    sentence = sentence.split(' ')
    sentence = [stemmer.stem(word) for word in sentence if word not in set(stopwords.words('english'))]
    while '' in sentence:
        sentence.remove('')
    sentence = ' '.join(sentence)
    text.append(sentence)
    return text

def tokenization(text):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1
    text_sequences = tokenizer.texts_to_sequences(text)
    padding_sequence = pad_sequences(text_sequences,maxlen=80)
    return padding_sequence

dt = "Whether you are in simulation or machine learning, you will find their combination offers powerful benefits for your developments."
a = preprocessing(dt)
final = tokenization(a)


model = load_model("D:\\vs code\python\DeepLearning\\NLP\projects\emailSpamDetector\model.h5")

result = model.predict(final) # the result is not in the binary format so we have to convert it into binary format
threshold = 0.5
binary_predictions = (result >= threshold).astype(int)

if binary_predictions[0][0] == 0:
    print('spam')
else:
    print('ham')