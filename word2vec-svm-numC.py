#! /usr/bin/env python
# -*-coding: utf-8-*-

from gensim.models import word2vec
from sklearn import cross_validation
from sklearn import svm
from sklearn.metrics import f1_score, confusion_matrix, make_scorer
import matplotlib.pyplot as plt
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

#matriz de confusion
confusionMatrix = numpy.array([[0,0,0],[0,0,0],[0,0,0]])
bestConfusionMatrix = numpy.array([[0,0,0],[0,0,0],[0,0,0]])
bestF1 = 0

#variables con los parametros de entrenamiento con word2vec
numCaracteristicas = 100
dimVentana = 4
minPalabras = 20
numCpus = 4
downsampling = 1e-3

#variables para dibujar grafico
f1 = numpy.zeros((10, ), dtype='float32')
desviacion = numpy.zeros((10, ), dtype='float32')
axis = [100,1000,0,100]
vx = [100,200,300,400,500,600,700,800,900,1000]

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
    return tweet

print('tokenizando tweets')
print('----------------------------------')

tweetsTokenizados = []
#crear lista con la clasificación de cada tweet
tweetClass = []
for tweet in tweets_categorizados:
    tweetsTokenizados.append(preparar_texto(tweet[1], eliminarStopwords, eliminarSimbolos))
    tweetClass.append(tweet[2])

#método para calcular el f1 y matriz de confusión de cada particion
def scoreF1_cm(y_true, y_pred):
    #cálculo del f1 como una media de f1s de tweets positivos y negativos
    f1_pos = f1_score(y_true, y_pred, pos_label='positive', average='macro')
    f1_neg = f1_score(y_true, y_pred, pos_label='negative', average='macro')
    #actualización de la matriz de confusión
    global confusionMatrix
    cfm = confusion_matrix(y_true, y_pred, labels=['positive', 'negative', 'neutral'])
    confusionMatrix += cfm
    return (f1_pos + f1_neg)/2

#método para crear una métrica de puntuación
def scorer():
    return make_scorer(scoreF1_cm, greater_is_better=True) 

#archivo donde se almacenan los resultados
textSave = open('word2vec_svmLineal_numC.txt', 'w')

#clasificar 10 veces
for i in range(10):
    print 'nCaracteristicas: '+str(numCaracteristicas)+'\tdimVentana: '+str(dimVentana)+'\tminPalabras: '+str(minPalabras)+'\n'

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

    print('10 crossfold validation - linear support vector machine')
    print('----------------------------------')
    confusionMatrix = numpy.array([[0,0,0],[0,0,0],[0,0,0]])

    clf = svm.SVC(kernel='linear', C=3, degree=1)
    scores = cross_validation.cross_val_score(clf, tweetMatrix, tweetClass, cv=10, scoring=scorer())
    print("F1: %0.2f (+/- %0.2f)" % (scores.mean()*100, scores.std() * 200))
    print confusionMatrix

    #almacenar resultados
    textSave.write('nCaracteristicas: '+str(numCaracteristicas)+'\tdimVentana: '+str(dimVentana)+'\tminPalabras: '+str(minPalabras)+'\n')
    textSave.write('F1: '+str(scores.mean()*100)+ ' (+/- '+str(scores.std()*200)+')'+'\n')
    textSave.write('Matriz de confusion:\n')
    textSave.write(str(confusionMatrix)+'\n')
    textSave.write('\n')

    #almacenar media y desviacion
    f1[i]=scores.mean()*100
    desviacion[i]=scores.std()*200

    #almacenar mejor matriz de confusion
    if scores.mean() > bestF1:
        bestConfusionMatrix = confusionMatrix
    
    #actualización de variables
    numCaracteristicas += 100
    #dimVentana += 1
    #minPalabras += 1

textSave.close()

#dibujar grafica de f-score
plt.errorbar(vx, f1, yerr=desviacion)
plt.axis(axis)
plt.ylabel('F-score')
plt.xlabel('Numero de caracteristicas')
plt.title('Variacion del f-score segun el numero de caracteristicas')
plt.show()

#normalizar matriz por filas
mcn = bestConfusionMatrix.astype('float')/bestConfusionMatrix.sum(axis=1)[:, numpy.newaxis]
tick_marks = [0,1,2]
tick_names = ['positivo','negativo','neutral']

#dibujar matriz de confusion
plt.imshow(mcn, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de confusion')
plt.colorbar()
plt.xticks(tick_marks, tick_names, rotation=45)
plt.yticks(tick_marks, tick_names)
plt.tight_layout()
plt.ylabel('Polaridad real')
plt.xlabel('Polaridad predicha')
plt.show()

