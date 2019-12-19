##Performs preprocessing for twitter-training data
from sklearn.preprocessing import LabelEncoder
import numpy as np
import gzip
import json
import math
from representation import parseJsonLine, Place, extractPreprocessUrl
import pickle
import string
from gensim.models.fasttext import FastText
from keras.preprocessing.sequence import pad_sequences


trainingFile="data/train/training.twitter.json.gz" #File with  all ~9 Million training tweets
placesFile='data/train/training.json.gz'           #Place annotation provided by task organisers
binaryPath= 'data/binaries/'            #Place to store the results

knownTweets = set()
with gzip.open(trainingFile,'rb') as file:
    for line in file:
        instance = parseJsonLine(line.decode('utf-8'))
        knownTweets.add(instance.id)


#Parse and add gold-label for tweets
tweetToTextMapping= {} # Map<Twitter-ID; place>
with gzip.open(placesFile,'rb') as file:
    for line in file:
        parsed_json = json.loads(line.decode('utf-8'))
        tweetId=int(parsed_json["tweet_id"])
        if(tweetId in knownTweets):
            place = Place(name=parsed_json["tweet_city"], lat=parsed_json["tweet_latitude"], lon=parsed_json["tweet_longitude"])
            tweetToTextMapping[tweetId] = place

print(str(len(tweetToTextMapping.keys())) + " tweets for training are found")
del(knownTweets)


###########Find the mean location for each of the ~3000 classes
placeSummary = {}
for place in list(map(lambda  x : x, tweetToTextMapping.values())):
    if place._name not in placeSummary:
        placeSummary[place._name] = []

    placeSummary[place._name].append(place)



#Calculate the mean lon/lat location for each city as city center
placeMedian = {}
for key in placeSummary.keys():
    lat = np.mean(list(map(lambda x: float(x._lat), placeSummary[key])))
    lon = np.mean(list(map(lambda x: float(x._lon), placeSummary[key])))
    placeMedian[key] = (lat,lon)
del(placeSummary)


classEncoder = LabelEncoder()
classEncoder.fit(list(placeMedian.keys()))




fastTextModel = "/media/philippe/5f695998-f5a5-4389-a2d8-4cf3ffa1288a/embeddings/crawl-300d-2M-subword/crawl-300d-2M-subword.bin"
model = FastText.load_fasttext_format(fastTextModel)


#2.) Tokenize texts
def my_filter():
    f = string.punctuation
    f += '\t\n\r…”'
    return f


translate_map = dict((ord(c), "") for c in my_filter())


###Extract relevant parts and store in list...
def getDocumentVectors(text, model):
    tokens = text.split()
    docRep = []
    for token in tokens:
        token = token.lower()
        token = token.translate(translate_map)
        vector = model[token]
        docRep.append(vector)
    return np.asarray(docRep)

""""
with gzip.open(trainingFile, 'rb') as file:
    trainTexts = [];
    trainLabels = []
    for line in file:

        instance = parseJsonLine(line.decode('utf-8'))

        trainTexts.append(getDocumentVectors(instance.text, model))

        trainLabel = tweetToTextMapping[instance.id]._name
        trainLabels.append(trainLabel)

        if len(trainTexts) % 1000000 == 0:
            print("Saving")
            filehandler = open(binaryPath +"trainData-" +str(len(trainTexts)) +".obj" , "wb")
            pickle.dump((trainTexts, trainLabels), filehandler)
            filehandler.close()

            trainTexts = []
            trainLabels = []
"""

def batch_generator(twitterFile, goldstandard, batch_size=64):
    while True: #TODO: needed?
        with gzip.open(twitterFile, 'rb') as file:
            trainTexts = []; trainLabels=[]
            for line in file:

                if len(trainTexts) == batch_size:
                    trainTexts = []; trainLabels = []

                instance = parseJsonLine(line.decode('utf-8'))

                trainTexts.append(getDocumentVectors(instance.text, model))

                trainLabel = goldstandard[instance.id]._name
                trainLabels.append(trainLabel)

                #print(str(instance.id) +"\t" +str(len(trainDescriptions)))

                if len(trainTexts) == batch_size:

                    #Text Tweet
                    trainTexts = np.asarray(trainTexts)  # Convert to ndArraytop
                    trainTexts = pad_sequences(trainTexts, dtype = 'float32')

                    # class label
                    classes = classEncoder.transform(trainLabels)

                    #yield trainDescriptions, classes
                    yield ({
                            'inputText' : trainTexts
                           },
                           classes
                           )

import time
from keras.layers import Dropout, Dense, BatchNormalization, SpatialDropout1D, LSTM, Input
from keras.layers.embeddings import Embedding
from keras import Model
import datetime

# create the model
batch_size = 256
nb_epoch = 5
verbosity=2
textEmbeddings = 300
numberOfTrainingsamples=9127900 #TODO zcat training.twitter.json.gz  | wc -l


#5.) Text Model
textBranchI = Input(shape=(None, textEmbeddings), name="inputText")
textBranch = SpatialDropout1D(rate=0.2)(textBranchI)
textBranch = BatchNormalization()(textBranch)
#textBranch = Dropout(0.2)(textBranch)
textBranch = LSTM(units=100, recurrent_dropout=0.2)(textBranch)
textBranch = BatchNormalization()(textBranch)
textBranch = Dropout(0.2, name="text")(textBranch)
textBranchO = Dense(len(classEncoder.classes_), activation='softmax')(textBranch)

textModel = Model(inputs=textBranchI, outputs=textBranchO)
textModel.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
textModel.summary()

start = time.time()
textHistory = textModel.fit_generator(generator=batch_generator(twitterFile=trainingFile, goldstandard=tweetToTextMapping, batch_size=batch_size),
                    epochs=nb_epoch, steps_per_epoch=math.ceil(numberOfTrainingsamples/batch_size),
                    verbose=verbosity
                    )
print("textBranch finished after " +str(datetime.timedelta(seconds=round(time.time() - start))))

binaryPath= 'data/binaries/'    #Place where the serialized training data is
modelPath= 'data/models/'       #Place to store the models

textModel.save(modelPath +'GeneratortextBranchNorm.h5')



#Map class int representation to
#colnames = [None]*len(set(classes))
#for i in range(len(classes)):
#    colnames[classes[i]] =  trainLabels[i]

filehandler = open(binaryPath + "GeneratorProcessors.obj", "wb")
pickle.dump(( placeMedian, classEncoder ), filehandler)
filehandler.close()
