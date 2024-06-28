import os
import json
import string
import random
import nltk #pour le traitement de texte 
import numpy as np
from nltk.stem import WordNetLemmatizer # raçinisation des mots 
import tensorflow as tf
from tensorflow.keras.models import load_model # importer le modèle en trainé
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
nltk.download("punkt")
nltk.download("wordnet")

# Construire le chemin complet vers le modèle
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) #retourner au chemin absolu du dossier parent 
model_path = os.path.join(project_root, 'models', 'simple_chatbot_gen.h5') 

model = load_model(model_path) # on charge le modèle 

lemmatizer = WordNetLemmatizer() # on initialise le lemmatizer

def vocabs(data):
    # initialisation de lemmatizer pour obtenir la racine des mots
    # création des listes
    words = []
    classes = []
    doc_X = []
    doc_y = []

    # parcourir avec une boucle For toutes les intentions
    # tokéniser chaque pattern et ajouter les tokens à la liste words, les patterns et
    # le tag associé à l'intention sont ajoutés aux listes correspondantes
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            tokens = nltk.word_tokenize(pattern)
            words.extend(tokens) # les tokens 
            doc_X.append(pattern) # doc contenant toutes les types de questions
            doc_y.append(intent["tag"]) # doc contenant tous les tags 

        # ajouter le tag aux classes s'il n'est pas déjà là
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

    # lemmatiser tous les mots du vocabulaire et les convertir en minuscule
    # si les mots n'apparaissent pas dans la ponctuation
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]

    # trier le vocabulaire et les classes par ordre alphabétique et prendre le
    # set pour s'assurer qu'il n'y a pas de doublons
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
    bow = bag_of_words(text, vocab) #vecteur de 0 et de 1 avec 1 pour les valeurs qui apparaissent dans le vocabulaire 
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
