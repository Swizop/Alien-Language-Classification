import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing, metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

f = open("alien_language/train_samples.txt", encoding="utf-8")
f = f.readlines()
g = open("alien_language/train_labels.txt", encoding="utf-8")
g = g.readlines()

countVectorizer = CountVectorizer(lowercase=False, binary=True, ngram_range=(3, 6), strip_accents=None, analyzer="char")

def preprocess(f, g, trainCall=True):
    """
        Pentru un fisier f cu fraze alien si un fisier g cu clasele lor, functia returneaza un tuplu
        (
            frazele transformate folosind countvectorizer,
            clasele carora apartin
        )
        trainCall == True => functia este apelata pentru fisierele de training. Altfel, inseamna ca este
            apelata pentru fisierele de validare. In primul caz, este modelat countvectorizer-ul in cadrul functiei
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
        trainSamples.append(lineSplit[1][:-1])      # \n-ul de la final e eliminat
        
        lineSplit = g[i].split("\t")
        trainLabelsIds.append(lineSplit[0])
        trainLabels.append(int(lineSplit[1][:-1]))
        i += 1

    trainLabels = np.array(trainLabels)
    trainSamples = np.array(trainSamples)
    
    if trainCall == True:
        countVectorizer.fit(trainSamples)
    trainSamples = countVectorizer.transform(trainSamples)
    
    return trainSamples, trainLabels


trainSamples, trainLabels = preprocess(f, g)
f = open("alien_language/validation_samples.txt", encoding="utf-8")
f = f.readlines()
g = open("alien_language/validation_labels.txt", encoding="utf-8")
g = g.readlines()
valSamples, valLabels = preprocess(f, g, False)


knModel = KNeighborsClassifier(n_neighbors=3, weights="distance", p = 1)
knModel.fit(trainSamples, trainLabels)
predictii = knModel.predict(valSamples)
print(knModel.score(valSamples, valLabels))
print(metrics.confusion_matrix(valLabels, predictii))