import nltk
import numpy as np 
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# from keras.layers import Dense, Activation, Dropout
# from keras import Sequential
# from keras.optimizers import SGD
# from keras.optimizers import Adam
import pickle


from sklearn.linear_model import LogisticRegression


dataset = pd.read_csv("D://Datasets//spam.csv", encoding='ISO-8859-1')

stemmer_1 = PorterStemmer()
stemmer_2 = SnowballStemmer('english')

vectorizer = CountVectorizer()
cVectorizer = TfidfVectorizer(max_features=1500)
encoder = LabelEncoder()

rows = len(dataset.axes[0])
columns = len(dataset.axes[1])


print("----------------------------------------------------------------------------------------------------")

dataset.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], inplace= True, axis=1)

# data preprocessing

dataset["length"] = dataset["v2"].apply(len)

print(dataset["length"].max())

corpus = []

for i in range(len(dataset['v2'])):
    review = re.sub('[^a-zA-Z]', ' ', dataset['v2'][i])
    review = review.lower()
    review = review.split()
    review = [stemmer_1.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = " ".join(review)
    corpus.append(review)


print("Preprocessing Completed")

# # Creating a bag of words model

# x = vectorizer.fit_transform(corpus).toarray()
# y = encoder.fit_transform(dataset['v1'])

# Using TF-IDF vectorizer for data creation

x = vectorizer.fit_transform(corpus).toarray()
y = encoder.fit_transform((dataset['v1']))

print(y)
print(x)
# Using Machine learning model to predict the category

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state= 42)

print("------------------------------------------------------------------------------------")

train_data = vectorizer.fit_transform(x_train)
test_data = vectorizer.transform(x_test)

# print("-------------------------------------------------------------------------------------")

# print(x_test)

# print("--------------------------------------------------------------------------------------")

# print(y_train)

# print("--------------------------------------------------------------------------------------")

# print(y_test)

print("---------------------------------------------------------------------------------------")

# print(len(x_test))

# Model Training 

# using Logistic Regression

lR = LogisticRegression()

lR.fit(train_data, test_data)

prediction = lR.predict(test_data)

proba_prediction = lR.predict_proba(test_data)

print(len(prediction))

print(prediction[0])

print(proba_prediction[0])

score = lR.score(x_test, y_test)

# print(score) # score (0.9763101220387652)
# # after using tf-idf vectorizer score (0.9569274946159368)

with open('model.pkl','wb') as file:
    pickle.dump(lR, file)

# USING A ANN

# model = Sequential()

# model.add(Dense(128, input_shape=(len(x[0]),),activation='relu',))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(128, activation = 'relu'))
# model.add(Dropout(0.5))
# model.add(Dense(2, activation='softmax'))

# model.summary()

# sgd = SGD(learning_rate=(0.001), momentum=0.9, nesterov=True, decay = 1e-6)

# model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# hist = model.fit(np.array(x_train),np.array(y_train), batch_size=10, epochs=1000, verbose=2)  # score (418/418 - 1s - loss: 3.2061e-04 - accuracy: 0.9998 - 686ms/epoch - 2ms/step)

# model.save('D:\\vs code\python\DeepLearning\models\\spamDetector.h5', hist)

# print(len(x[0]))