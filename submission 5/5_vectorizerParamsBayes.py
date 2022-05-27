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
nbModel.fit(trainSamples, trainLabels)
nbModel.predict(valSamples)
print(nbModel.score(valSamples, valLabels))

# lowercase = False. 20k => 0.6872; 10k => 0.6848 ; 15k => 0.6876 ; 12k => 0.6874 ; 17k => 0.689 ; 16k => 0.6898
    # concluzie: 16k cel mai bun pe lowercase False. pe True e putin sub 20k

# max_df = 0.75 => mai prost
# binary == True => 0.6926 pe 20k + lowercase true si 0.6908 pe 16k + FAlse
# min_df => rezultate foarte proaste, ~ 0.4 pentru df 0.2 ; ~0.667 pt 0.001; cu cat e mai mic cu atat e mai mare acuratetea. prost si pt binary False
# ngram (1, 2) ; binary true; 20k + lowercase => 0.6958
    # .679 pt (2, 2) , .661 pt (2, 3) , .69 pt (1, 3)

# 0.6926 pare sa fie maximul pt abordare cu max features
# scoatem max features si .toarray  => 0.71 cu (1, 3) si 0.7134 cu (1, 2)
# lowercase false => 0.7122
# 0.7126 pentru max_df = 0.9
# lowercase=False, binary=False, ngram_range=(1, 2), max_df=0.9 => 0.7146
    # ascii strip accents => 0.7142 ; unicode => 0.714
    # binary = True => 0.716

# incercam parametri noi
# avem maximul 0.716 pentru analyzer pe cuvinte. incercam acum pe char
# (lowercase=False, binary=True, ngram_range=(1, 2), max_df=0.8, strip_accents="ascii", analyzer="char") => 0.5768, neimpresionant
    # schimbam ngram range-ul sa ia mai multe valori, are sens sa ia secvente mai lungi de litere.
    # (1, 4) => 0.7098 (2, 5) => 0.7244 (3, 6) => 0.7308 (4, 6) => 0.7332; (4, 7) => 0.7328
    # schimbam max df in 0.9 si e fix la fel, daca scoatem de tot e 0.733. 
    # umblam la chestii ce tin de litere: lowercase=True, 0.7322 . pastram False. strip -> unicode => 0.744 . None => 0.745
    # mai incercam sa schimbam nrange-ul. (3, 6) => 0.7478 (3, 5) => 0.7456 (2, 5) => 0.7468. daca facem intervalul prea mic riscam overfitting