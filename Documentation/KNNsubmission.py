import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing, metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

f = open("../alien_language/train_samples.txt", encoding="utf-8")
f = f.readlines()
g = open("../alien_language/train_labels.txt", encoding="utf-8")
g = g.readlines()
countVectorizer = CountVectorizer(lowercase=False, binary=True, ngram_range=(3, 6), strip_accents=None, analyzer="char")

def preprocess(f, g):
    """
        Functia primeste 2 fisiere txt de intrare si returneaza un tuplu de forma:
        (
            reprezentarea frazelor de antrenare in forma data de un countvectorizer,
            np array cu clasele lor, in ordinea citirii
        )
    """
    global countVectorizer
    trainSamplesIds = []
    trainSamples = []
    trainLabelsIds = []
    trainLabels = []
    i = 0
    for line in f:
        lineSplit = line.split("\t")        # => id-ul pe pozitia 0 si fraza pe pozitia 1
        trainSamplesIds.append(lineSplit[0])
        trainSamples.append(lineSplit[1][:-1])      # pe ultima pozitie a string-ului e \n, deci il taiem
        
        lineSplit = g[i].split("\t")
        trainLabelsIds.append(lineSplit[0])
        trainLabels.append(int(lineSplit[1][:-1]))
        i += 1

    trainLabels = np.array(trainLabels)
    trainSamples = np.array(trainSamples)
    
    countVectorizer.fit(trainSamples)
    trainSamples = countVectorizer.transform(trainSamples)
    
    return trainSamples, trainLabels

def preprocess_test(f):
    """
        Functia primeste un fisier txt de intrare care contine propozitiile si id-urile pt test.
        Returneaza un tuplu de forma:
        (
            frazele in forma data de countvectorizer antrenat pe datele de train,
            o lista simpla cu id-urile lor, necesare pt o ulterioara scriere in fisier de output
        )
    """
    global countVectorizer
    sampleIds = []
    samples = []
    
    for line in f:
        lineSplit = line.split("\t")
        sampleIds.append(lineSplit[0])
        samples.append(lineSplit[1][:-1])       # pe ultima pozitie e \n, il eliminam

    samples = np.array(samples)
    
    samples = countVectorizer.transform(samples)
    
    return samples, sampleIds

trainSamples, trainLabels = preprocess(f, g)
f = open("../alien_language/test_samples.txt", encoding="utf-8")
f = f.readlines()
testSamples, sampleIds = preprocess_test(f)


knModel = KNeighborsClassifier(n_neighbors=2, weights="distance", p = 1)
knModel.fit(trainSamples, trainLabels)
predictions = knModel.predict(testSamples)
g = open("../output/vectorizeKNNValidSet.txt", 'w')
g.write("id,label\n")
for i in range(len(predictions)):
    g.write(f"{sampleIds[i]},{predictions[i]}\n")
g.close()