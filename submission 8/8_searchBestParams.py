import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing, metrics
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB

# trainLabels = np.loadtxt('alien_language/train_samples.txt', delimiter='\t', dtype='str', encoding='utf-8')
# print(trainLabels[1][30])
# trainSamples = np.loadtxt('alien_language/train_labels.txt')
f = open("alien_language/train_samples.txt", encoding="utf-8")
f = f.readlines()
g = open("alien_language/train_labels.txt", encoding="utf-8")
g = g.readlines()
#countVectorizer = CountVectorizer(lowercase=False, binary=True, ngram_range=(1, 2), max_df=0.8, strip_accents="ascii")
countVectorizer = CountVectorizer(lowercase=False, binary=True, ngram_range=(3, 6), strip_accents=None, analyzer="char")
# countVectorizer = CountVectorizer(lowercase=False, binary=True, ngram_range=(2, 5), analyzer='char')

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

# print(type(trainSamples))
# print(trainSamples[0].sum())
# print(trainSamples[0].sum())

trainSamples, trainLabels = preprocess_and_normalize(f, g)
f = open("alien_language/validation_samples.txt", encoding="utf-8")
f = f.readlines()
g = open("alien_language/validation_labels.txt", encoding="utf-8")
g = g.readlines()
valSamples, valLabels = preprocess_and_normalize(f, g, False)


nbModel = MultinomialNB()
parametriDict = {'alpha': [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1, 1.1, 1.2, 1.5, 1.7, 2], 'fit_prior': [True, False]}
paramTuner = GridSearchCV(nbModel, parametriDict)
paramTuner.fit(trainSamples, trainLabels)
paramTuner.predict(valSamples)
print(paramTuner.score(valSamples, valLabels))
print(paramTuner.best_params_)      # => 0.7632 pentru alpha 0.1 si fit_prior True
# nbModel.predict(valSamples)
# (nbModel.score(valSamples, valLabels))