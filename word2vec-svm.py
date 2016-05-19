#! /usr/bin/env python
# -*-coding: utf-8-*-

from gensim.models import word2vec
from sklearn import cross_validation
from sklearn import svm
from nltk.corpus import stopwords
import nltk.data
import re
import os
import json
import numpy
import logging

numpy.seterr(divide='ignore', invalid='ignore')

#variable para separar parrafos de tweet
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

#variables para tener en cuenta stopwords y simbolos 
eliminarStopwords = True
eliminarSimbolos = True

#variables con los parametros de entrenamiento con word2vec
numCaracteristicas = 100
dimVentana = 4
minPalabras = 5
numCpus = 4
downsampling = 1e-3

#informacion de los procesos que realizan los modulos de Gensim
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

print('extrayendo tweets con su categoria')
print('----------------------------------')

#crear diccionario de twits con su categoria dic[id twit] -> categoria
f=open('ALC_corpus/twitter-train-cleansed-B.tsv')
d=f.read()
d=d.split('\n')
c={}
for i in d:
    if i!='':
        twit_c=i.split('\t')
        c[twit_c[0]] = twit_c[2]

f.close()

#crear lista de tweets con informacion de id, texto y su polaridad
tweets_categorizados = []       #lista de tuplas (tweet id, texto, polaridad)
archivos = os.listdir('ALC_corpus')
#twits = []
for i in archivos:
    if i[-5:]=='.json':   
        f=json.load(open('ALC_corpus/'+i)) #abrir y organizar archivo
        twit_id=f['id'] #id de twieet
        twit_text=f['text'] #texto de tweet
        tweets_categorizados.append((twit_id, twit_text, c[str(twit_id)])) #int, string, string

#Metodo para devolver una lista de texto en minusculas teniendo en cuenta o no los simbolos y stopwords
def preparar_frase(frase, useStopwords, useSimbolos):
    if useSimbolos:
        frase = re.sub("[^a-zA-Z]"," ", frase)
    words = frase.lower().split()
    if useStopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return(words)

#Metodo para devolver una lista de listas texto a partir de un grupo de parrafos
def preparar_texto(texto, useStopwords, useSimbolos):
    frases = tokenizer.tokenize(texto.strip())
    tweet = []
    for frase in frases:
        if len(frase) > 0:
            tweet += preparar_frase(frase, useStopwords, useSimbolos)
            #tweet.append(preparar_frase(frase, useStopwords, useSimbolos))
    return tweet

print('tokenizando tweets')
print('----------------------------------')

tweetsTokenizados = []
#crear lista con la clasificación de cada tweet
tweetClass = []
for tweet in tweets_categorizados:
    #tweetsTokenizados += preparar_texto(tweet[1], eliminarStopwords, eliminarSimbolos)
    tweetsTokenizados.append(preparar_texto(tweet[1], eliminarStopwords, eliminarSimbolos))
    tweetClass.append(tweet[2])

print('entrenando modelo word2vec')
print('----------------------------------')

#entrenar modelo
modelo = word2vec.Word2Vec(tweetsTokenizados, workers=numCpus, size=numCaracteristicas, min_count=minPalabras, window=dimVentana, sample=downsampling)

print('transformando tweets a vectores')
print('----------------------------------')

#crear matriz de caracteristicas con cada tweet
tweetMatrix = numpy.zeros((len(tweetsTokenizados), numCaracteristicas), dtype="float32")
vocabulario = set(modelo.index2word)
c = 0
for document in tweetsTokenizados:
    tweetVector = numpy.zeros((numCaracteristicas, ), dtype="float32")
    w = 0
    for word in document:
        if word in vocabulario:
            tweetVector = numpy.add(tweetVector, modelo[word])
            w += 1
    tweetVector = numpy.divide(tweetVector, w)
    #el vector del tweet se representa como una media del sumatorio de los vectores de cada una de sus palabras
    tweetMatrix[c] = tweetVector
    c += 1

#print 'tweetMatrix: '+str(c)
#print 'tweetClass: '+str(len(tweetClass))

print('10 crossfold validation - support vector machine')
print('----------------------------------')

clf = svm.SVC(kernel='linear', C=2)
scores = cross_validation.cross_val_score(clf, tweetMatrix, tweetClass, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

