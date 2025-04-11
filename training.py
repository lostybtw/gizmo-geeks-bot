# Imports
import numpy as np
import pickle
import json
import random

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer() # Lemmatizer object

intents = json.loads(open("intents.json").read()) # Get intents.json as a dictionary

# Array definitions
words = []
classes = []
documents = []
ignore_letters = ["?","!",",","."]

# words[] = each word in intents/patterns, documents[] = [words[i],intent tag, classes[], classes[] = tags 
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append([word_list,intent['tag']])
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize the words if word is not in ignore_letters
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
# Sort the words[] and classes[] and remove duplicates
words = sorted(set(words))
classes = sorted(set(classes))

# Dump Words, Classes To a File.
pickle.dump(words,open("words.pkl",'wb'))
pickle.dump(classes,open("classes.pkl",'wb'))

training = [] # training data  
output_empty = [0] * len(classes) # Empty array of all tags

for doc in documents:
    bag = [] # bag of words method 
    word_patterns = doc[0] 
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns] # word_patterns are words occuring in each tag
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0) # Create a bag of words
    output_row = list(output_empty) 
    output_row[classes.index(doc[1])] = 1
    training.append([bag,output_row]) # Training Data

# Randomly Shuffle training[] so order doesn't affect learning
random.shuffle(training)
# Convert Training to numpy array, important format fr ML
training = np.array(training,dtype=object)

# Divide in train_x -> Bag, train_y -> Labels
train_x = list(training[:,0])
train_y = list(training[:,1])

# Create Sequential Model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),)))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),))
model.add(Activation("softmax"))

sgd = SGD(lr=0.01,momentum=0.9,nesterov=True) # SGD Optimizer
model.compile(loss="categorical_crossentropy",optimizer=sgd,metrics=['accuracy']) # Compile model
hist = model.fit(np.array(train_x),np.array(train_y), epochs=200, batch_size=1, verbose=1) # Train model and save it to variable
print(hist)
model.save("chatbot2.h5",hist) # Save model to file
print("Trained")
