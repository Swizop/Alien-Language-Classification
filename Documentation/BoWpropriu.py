import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing, metrics
from sklearn.naive_bayes import MultinomialNB

f = open("../alien_language/train_samples.txt", encoding="utf-8")
f = f.readlines()
g = open("../alien_language/train_labels.txt", encoding="utf-8")
g = g.readlines()

vocabular = dict()          # cheia k apare in dictionar daca k apare in fisierul de train
listaVocabular = []         # lista cu cuvintele distincte din text, in ordinea aparitiei lor
                        # valoarea corespunzatoare lui k in dictionar va reprezenta pe ce pozitie se gaseste 
                            # cuvantul k in listaVocabular

def preprocess_and_normalize(f, g):
    global vocabular, listaVocabular

    trainSamplesIds = []
    trainSamples = []
    trainLabelsIds = []
    trainLabels = []
    i = 0
    for line in f:
        lineSplit = line.split("\t")
        trainSamplesIds.append(lineSplit[0])
        trainSamples.append(lineSplit[1][:-1])      # taiem \n-ul de pe ultima pozitie

        features = [0] * len(listaVocabular)        # contine atat de multe feature-uri cate s-au descoperit pana acum
        
        for word in trainSamples[-1].split(" "):        # trecem prin cuvintele celei mai recente fraze procesate
            if word not in vocabular:
                vocabular[word] = len(listaVocabular)
                listaVocabular.append(word)
                features.append(1)           # este un cuvant nou, deci lista de features trebuie extinsa. 
                                                # cuvantul are frecventa 1 pana acum in obiectul curent
            else:
                features[vocabular[word]] += 1
            
        trainSamples[-1] = features        # train samples va pastra propozitiile in forma post-procesare

        lineSplit = g[i].split("\t")
        trainLabelsIds.append(lineSplit[0])
        trainLabels.append(int(lineSplit[1][:-1]))
        i += 1

    trainLabels = np.array(trainLabels)

    # mai parcurgem o data frazele, pentru ca s-ar putea ca nu toate frazele sa aiba acelasi nr de feature-uri
    # cand antrenam, unele liste de cuvinte parcurse mai devreme s-ar putea sa ramana mai mici, ptc.
            # lipsesc cuvinte descoperite mai tarziu dar care ar fi trebuit reprezentate si pt. ele, cu valoarea 0
    for i in range(len(trainSamples)):
        if len(trainSamples[i]) < len(listaVocabular):      
            trainSamples[i].extend([0 for _ in range(len(listaVocabular) - len(trainSamples[i]))])

        trainSamples[i] = np.array(trainSamples[i])
    trainSamples = np.array(trainSamples)
    
    
    return trainSamples, trainLabels

def preprocess_and_normalize_test(f):
    global vocabular, listaVocabular
    sampleIds = []
    samples = []
    
    for line in f:
        lineSplit = line.split("\t")
        sampleIds.append(lineSplit[0])
        samples.append(lineSplit[1][:-1])

        # ne intereseaza doar cuvintele pe care le-am descoperit deja in train
        features = [0] * len(listaVocabular)
        for word in samples[-1].split(" "):
            if word in vocabular:
                features[vocabular[word]] += 1
        samples[-1] = np.array(features)

    samples = np.array(samples)
    
    return samples, sampleIds

trainSamples, trainLabels = preprocess_and_normalize(f, g)
f = open("../alien_language/test_samples.txt", encoding="utf-8")
f = f.readlines()
testSamples, sampleIds = preprocess_and_normalize_test(f)


nbModel = MultinomialNB()
nbModel.fit(trainSamples, trainLabels)
predictions = nbModel.predict(testSamples)
print(len(predictions))
print(len(sampleIds))
g = open("vectorizebayes.txt", 'w')
g.write("id,label\n")
for i in range(len(predictions)):
    g.write(f"{sampleIds[i]},{predictions[i]}\n")
g.close()