import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing, metrics
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB

# trainLabels = np.loadtxt('alien_language/train_samples.txt', delimiter='\t', dtype='str', encoding='utf-8')
# print(trainLabels[1][30])
# trainSamples = np.loadtxt('alien_language/train_labels.txt')
f = open("../alien_language/train_samples.txt", encoding="utf-8")
f = f.readlines()
g = open("../alien_language/train_labels.txt", encoding="utf-8")
g = g.readlines()
countVectorizer = CountVectorizer(lowercase=False, binary=True, ngram_range=(3, 6), strip_accents=None, analyzer="char")

def preprocess_and_normalize(f, g, trainCall=True):
    global countVectorizer
    trainSamplesIds = []
    trainSamples = []
    trainLabelsIds = []
    trainLabels = []
    i = 0
    for line in f:
        lineSplit = line.split("\t")
        trainSamplesIds.append(lineSplit[0])
        trainSamples.append(lineSplit[1][:-1])
        
        lineSplit = g[i].split("\t")
        trainLabelsIds.append(lineSplit[0])
        trainLabels.append(int(lineSplit[1][:-1]))
        i += 1

    trainLabels = np.array(trainLabels)
    trainSamples = np.array(trainSamples)
    
    if trainCall == True:
        # print(trainSamples.shape)
        countVectorizer.fit(trainSamples)
    trainSamples = countVectorizer.transform(trainSamples)
    # trainSamples = preprocessing.normalize(trainSamples, norm='l2')
    
    return trainSamples, trainLabels

def preprocess_and_normalize_test(f):
    global countVectorizer
    sampleIds = []
    samples = []
    
    for line in f:
        lineSplit = line.split("\t")
        sampleIds.append(lineSplit[0])
        samples.append(lineSplit[1][:-1])

    samples = np.array(samples)
    
    samples = countVectorizer.transform(samples)
    # trainSamples = preprocessing.normalize(trainSamples, norm='l2')
    
    return samples, sampleIds

# print(type(trainSamples))
# print(trainSamples[0].sum())
# print(trainSamples[0].sum())

f2 = open("../alien_language/validation_samples.txt", encoding="utf-8").readlines()
g2 = open("../alien_language/validation_labels.txt", encoding="utf-8").readlines()
f.extend(f2)
g.extend(g2)

trainSamples, trainLabels = preprocess_and_normalize(f, g)
f = open("../alien_language/test_samples.txt", encoding="utf-8")
f = f.readlines()
testSamples, sampleIds = preprocess_and_normalize_test(f)


nbModel = MultinomialNB()
parametriDict = {'alpha': [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1, 1.1, 1.2, 1.5, 1.7, 2], 'fit_prior': [True, False]}
paramTuner = GridSearchCV(nbModel, parametriDict)
paramTuner.fit(trainSamples, trainLabels)
print(paramTuner.best_params_)      # fit prior False cand se antreneaza cu tot cu validation data. cu fit prior False setat manual in fisierul de train, aveam 0.7626
predictions = paramTuner.predict(testSamples)
g = open("../output/gridSearchParams.txt", 'w')
g.write("id,label\n")
for i in range(len(predictions)):
    g.write(f"{sampleIds[i]},{predictions[i]}\n")
g.close()