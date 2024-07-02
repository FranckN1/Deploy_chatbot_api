# functions.py

import os
import json
import string
import random
import nltk # pour le traitement de texte 
import numpy as np
from nltk.stem import WordNetLemmatizer # raçinisation des mots 
import tensorflow as tf
from tensorflow.keras.models import load_model # importer le modèle en trainé
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

nltk.download("punkt")
nltk.download("wordnet")


# Définir l'optimiseur standard
custom_optimizer = Adam(learning_rate=0.001)

# Construire le chemin complet vers le modèle
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # retourner au chemin absolu du dossier parent 
model_path = os.path.join(project_root, 'models', 'chatbot_gen.h5')

# Charger le modèle avec les objets personnalisés
model = load_model(model_path, custom_objects={'Adam': custom_optimizer})

lemmatizer = WordNetLemmatizer() # on initialise le lemmatizer

def vocabs(data):
    words = []
    classes = []
    doc_X = []
    doc_y = []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            tokens = nltk.word_tokenize(pattern)
            words.extend(tokens) # les tokens 
            doc_X.append(pattern) # doc contenant toutes les types de questions
            doc_y.append(intent["tag"]) # doc contenant tous les tags 
        if intent["tag"] not in classes:
            classes.append(intent["tag"])
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]
    words = sorted(set(words))
    classes = sorted(set(classes))
    return words, classes, doc_X, doc_y

def clean_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def bag_of_words(text, vocab):
    tokens = clean_text(text)
    bow = [0] * len(vocab) # vecteur de taille du vocabulaire (words)
    for w in tokens: # on parcour le message reçu
        for idx, word in enumerate(vocab): #on recup le mot et l'index
            if word == w: # Chaque élément du vecteur représente la présence ou l'absence de mots du vocabulaire dans le texte d'entrée.
                bow[idx] = 1
    return np.array(bow)

def pred_class(text, vocab, labels):
    bow = bag_of_words(text, vocab) # vecteur de 0 et de 1 avec 1 pour les valeurs qui apparaissent dans le vocabulaire 
    result = model.predict(np.array([bow]))[0]
    thresh = 0.2
    y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]
    y_pred.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in y_pred:
        return_list.append(labels[r[0]])
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result
