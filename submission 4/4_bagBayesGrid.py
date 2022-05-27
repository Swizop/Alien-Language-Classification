import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing, metrics
from sklearn.naive_bayes import MultinomialNB

# trainLabels = np.loadtxt('alien_language/train_samples.txt', delimiter='\t', dtype='str', encoding='utf-8')
# print(trainLabels[1][30])
# trainSamples = np.loadtxt('alien_language/train_labels.txt')
f = open("alien_language/train_samples.txt", encoding="utf-8")
f = f.readlines()
g = open("alien_language/train_labels.txt", encoding="utf-8")
g = g.readlines()

vocabular = dict()
listaVocabular = []

def preprocess_and_normalize(f, g, trainCall=True):
    global vocabular, listaVocabular

    trainSamplesIds = []
    trainSamples = []
    trainLabelsIds = []
    trainLabels = []
    i = 0
    for line in f:
        lineSplit = line.split("\t")
        trainSamplesIds.append(lineSplit[0])
        trainSamples.append(lineSplit[1][:-1])

        features = [0] * len(listaVocabular)
        
        for word in trainSamples[-1].split(" "):
            if word not in vocabular:
                if trainCall == True:       # daca avem apel pt setul de training, trb generat vocabularul:
                    vocabular[word] = len(listaVocabular)
                    listaVocabular.append(word)
                    features.append(1)
            else:
                features[vocabular[word]] += 1
            
        trainSamples[-1] = features

        lineSplit = g[i].split("\t")
        trainLabelsIds.append(lineSplit[0])
        trainLabels.append(int(lineSplit[1][:-1]))
        i += 1

    trainLabels = np.array(trainLabels)

    for i in range(len(trainSamples)):
        if len(trainSamples[i]) < len(listaVocabular):      # cand antrenam, unele liste de cuvinte s-ar putea sa fie mai mici, ptc. lipsesc cuvinte descoperite mai tarziu
            trainSamples[i].extend([0 for _ in range(len(listaVocabular) - len(trainSamples[i]))])

        trainSamples[i] = np.array(trainSamples[i])
    trainSamples = np.array(trainSamples)
    
    
    return trainSamples, trainLabels

# print(type(trainSamples))
# print(trainSamples[0].sum())
# print(trainSamples[0].sum())

trainSamples, trainLabels = preprocess_and_normalize(f, g)
f = open("alien_language/validation_samples.txt", encoding="utf-8")
f = f.readlines()
g = open("alien_language/validation_labels.txt", encoding="utf-8")
g = g.readlines()
valSamples, valLabels = preprocess_and_normalize(f, g, False)


nbModel = MultinomialNB(fit_prior=False)
nbModel.fit(trainSamples, trainLabels)
nbModel.predict(valSamples)
print(nbModel.score(valSamples, valLabels))
# predictions = nbModel.predict(valSamples)
# print(metrics.accuracy_score(valLabels, predictions))

# concluzii : fit_prior == False face ca probabilitatea ca un  element sa fie in clasa sa nu fie invatata,
    # sa se foloseasca o functie gen f(o) = 1/3 pt toate obiectele. evitam overfittingul
# simplu: 0.7304 . fit_prior false => 0.732
# false & 0.1 => 0.722; 0.5 => 0.7282 ; 2 -> 0.715 ; 0.01 -> 0.7118 ; 0005 -> 0.707