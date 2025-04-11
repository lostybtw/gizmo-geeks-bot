# importing stuff
import numpy as np
import pickle
import json
import random

# Lemmatizer to simplify words
import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model # function to load the trained model

lemmatizer = WordNetLemmatizer() # creating lemmatizer object

intents = json.loads(open("intents.json").read())

# Loading words classes(tags) and model
words = pickle.load(open("words.pkl","rb"))
classes = pickle.load(open("classes.pkl","rb"))
model = load_model("chatbot2.h5")

# Clean a sentence into word tokens and lemmatize them
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Use bag_of_words method to get the word predicted
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words) # Empty bag of words
    for w in sentence_words: # iterate through the word list
        for i,word in enumerate(words): 
            if w == word:
                bag[i] = 1
    return np.array(bag) # Numpy Array bag

def predict_class(sentence):
    bow = bag_of_words(sentence) # Make bag of words
    res = model.predict(np.array([bow]))[0] # predict result
    ERROR_THRESHOLD = 0.4 # margin of error
    results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD] # create list of results in numbers and probability
    results.sort(key=lambda x:x[1], reverse=True) # sort results according to highest probability
    return_list = [] 
    for r in results:
        return_list.append({"intent":classes[r[0]],"probability":str(r[1])}) # replace result index with tags
    return return_list

def get_response(intents_list,intents_json):
    tag = intents_list[0]['intent'] # get the tag of result with best probability
    list_of_intents = intents_json['intents'] # find the list of tags in intents.json
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses']) # return a response randomly
            break
    return result

def run():
    message = input(" >_ You: ") # get user message
    ints = predict_class(message) # predict tag from message
    res = get_response(ints,intents) # pick a response
    print(" %% Bot: ",res) # print response
    if ints[0]["intent"] == "goodbye":
        print("---------------------------------------------- The Program Ends Here. ----------------------------------------------")
        exit() # End the Program

print("---------------------------------------------- Hello, World. I am A bot. ----------------------------------------------")
while True:
    run()
