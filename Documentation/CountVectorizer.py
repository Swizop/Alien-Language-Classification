import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing, metrics
from sklearn.naive_bayes import MultinomialNB

f = open("alien_language/train_samples.txt", encoding="utf-8")
f = f.readlines()
g = open("alien_language/train_labels.txt", encoding="utf-8")
g = g.readlines()
countVectorizer = CountVectorizer(lowercase=False, binary=True, ngram_range=(3, 6), strip_accents=None, analyzer="char")


def preprocess_and_normalize(f, g, trainCall=True):
    """
        Functie folosita pentru a configura reprezentarea atat datelor de train, cat si celor 
        de validare. Daca functia se va folosi pt. datele de validare, se seteaza trainCall la False.
    """
    global countVectorizer
    trainSamplesIds = []
    trainSamples = []
    trainLabelsIds = []
    trainLabels = []
    i = 0
    for line in f:
        lineSplit = line.split("\t")
        trainSamplesIds.append(lineSplit[0])
        trainSamples.append(lineSplit[1][:-1])      # eliminam \n-ul
        
        lineSplit = g[i].split("\t")
        trainLabelsIds.append(lineSplit[0])
        trainLabels.append(int(lineSplit[1][:-1]))
        i += 1

    trainLabels = np.array(trainLabels)
    trainSamples = np.array(trainSamples)
    
    if trainCall == True:       # pentru datele de antrenare va trebui sa configuram bag of words-ul
        countVectorizer.fit(trainSamples)
    trainSamples = countVectorizer.transform(trainSamples)
    
    return trainSamples, trainLabels


trainSamples, trainLabels = preprocess_and_normalize(f, g)
f = open("alien_language/validation_samples.txt", encoding="utf-8")
f = f.readlines()
g = open("alien_language/validation_labels.txt", encoding="utf-8")
g = g.readlines()
valSamples, valLabels = preprocess_and_normalize(f, g, False)


nbModel = MultinomialNB()
nbModel.fit(trainSamples, trainLabels)
nbModel.predict(valSamples)
print(nbModel.score(valSamples, valLabels))
