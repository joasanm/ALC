#! /usr/bin/env python
# -*-coding: utf-8-*-

from gensim.models import word2vec
from nltk.corpus import stopwords
import nltk.data
import re
import os
import json
import numpy
import logging

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
print('---------------------------------')

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
        #user_id=f['user']['id'] #id de usuario
        twit_text=f['text'] #texto de tweet
        tweets_categorizados.append((twit_id, twit_text, c[str(twit_id)])) #int, string, string
        #twits.append([twit_id, user_id, twit_text])


print('tweets extraidos con su categoria')
print('---------------------------------')

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
            #tweet += preparar_frase(frase, useStopwords, useSimbolos)
            tweet.append(preparar_frase(frase, useStopwords, useSimbolos))
    return tweet


print('----------10 cross fold----------')
print('---------------------------------')

particion = len(tweets_categorizados)//10

for i in range(10):
    #crear particiones train y test
    print('particion ' + str(i+1) + ' de 10')
    if i==0:
        train = tweets_categorizados[particion:]
        test = tweets_categorizados[:particion]
    elif i==9:
        train = tweets_categorizados[:particion*i]
        test = tweets_categorizados[particion*i:]
    else:
        train = tweets_categorizados[:particion*i]+tweets_categorizados[particion*(i+1):]
        test = tweets_categorizados[particion*i:particion*(i+1)]
    
    #crear lista de tweets procesados del train
    documentsTrain = []
    for j in train:
        #documents.append(preparar_texto(j[1], eliminarStopwords, eliminarSimbolos))
        documentsTrain += preparar_texto(j[1], eliminarStopwords, eliminarSimbolos)

    #crear lista de tweets procesados del test
    documentsTest = []
    for j in test:
        #documents.append(preparar_texto(j[1], eliminarStopwords, eliminarSimbolos))
        documentsTest += preparar_texto(j[1], eliminarStopwords, eliminarSimbolos)

    #nombre del archivo donde se guarda el modelo entrenado
    nombre_modelo = "p"+str(i+1)+"_nc_"+str(numCaracteristicas)+"_dv_"+str(dimVentana)+"_mp_"+str(minPalabras)
    print nombre_modelo

    #entrenar modelo
    modelo = word2vec.Word2Vec(documentsTrain, workers=numCpus, size=numCaracteristicas, min_count=minPalabras, window=dimVentana, sample=downsampling)

    #guardar modelo
    modelo.init_sims(replace=False)
    modelo.save(nombre_modelo)

    #crear matriz de caracteristicas con cada tweet del train
    trainVectors = numpy.zeros((len(documentsTrain), numCaracteristicas), dtype="float32")
    vocabulario = set(modelo.index2word)
    #maxWords = 0
    c = 0
    for document in documentsTrain:
        tweetVector = numpy.zeros((numCaracteristicas, ), dtype="float32")
        #w = 0
        for word in document:
            if word in vocabulario:
                tweetVector = numpy.add(tweetVector, modelo[word])
                #w += 1
        #el vector del tweet se representa como un sumatorio de los vectores de cada una de sus palabras
        trainVectors[c] = tweetVector
        #if maxWords < w:
            #maxWords = w
        c += 1

    #crear matriz de caracteristicas con cada tweet del test
    testVectors = numpy.zeros((len(documentsTest), numCaracteristicas), dtype="float32")
    c = 0
    for document in documentsTest:
        tweetVector = numpy.zeros((numCaracteristicas, ), dtype="float32")
        #w = 0
        for word in document:
            if word in vocabulario:
                tweetVector = numpy.add(tweetVector, modelo[word])
                #w += 1
        #el vector del tweet se representa como un sumatorio de los vectores de cada una de sus palabras
        testVectors[c] = tweetVector
        #if maxWords < w:
            #maxWords = w
        c += 1


