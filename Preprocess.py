##Performs preprocessing for twitter-training data

import numpy as np
import gzip
import json
from representation import parseJsonLine, Place, extractPreprocessUrl

trainingFile="data/train/training.twitter.json.gz" #File with  all ~9 Million training tweets
placesFile='data/train/training.json.gz'           #Place annotation provided by task organisers

#Parser Twitter-JSON
tweetToTextMapping= {} # Map<Twitter-ID; tweet>
with gzip.open(trainingFile,'rb') as file:
    for line in file:
        instance = parseJsonLine(line.decode('utf-8'))
        tweetToTextMapping[instance.id] = instance

#Add gold-label for tweets
with gzip.open(placesFile,'rb') as file:
    for line in file:
        parsed_json = json.loads(line.decode('utf-8'))
        tweetId=int(parsed_json["tweet_id"])
        if(tweetId in tweetToTextMapping):
            place = Place(name=parsed_json["tweet_city"], lat=parsed_json["tweet_latitude"], lon=parsed_json["tweet_longitude"])
            tweetToTextMapping[tweetId].place = place

print(str(len(tweetToTextMapping.keys())) + " tweets for training are found")

#city=[]
#with gzip.open(placesFile,'rb') as file:
#    for line in file:
#        parsed_json = json.loads(line.decode('utf-8'))
#        tweetId=parsed_json["tweet_city"]
#        city.append(tweetId)

###########Find the mean location for each of the ~3000 classes
placeSummary = {}
for place in list(map(lambda  x : x.place, tweetToTextMapping.values())):
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

###Extract relevant parts and store in list...
trainLabels = []  # list of label ids

trainDescription=[]
trainLinks=[]
trainLocation=[]
trainSource=[]
trainTexts = []
trainUserName=[]
trainTZ=[]
trainUtc=[]
for key in tweetToTextMapping:
    trainLabels.append(tweetToTextMapping[key].place._name)

    trainDescription.append(str(tweetToTextMapping[key].description))
    trainLinks.append(extractPreprocessUrl(tweetToTextMapping[key].urls))
    trainLocation.append(str(tweetToTextMapping[key].location))
    trainSource.append(str(tweetToTextMapping[key].source))
    trainTexts.append(tweetToTextMapping[key].text)
    trainUserName.append(str(tweetToTextMapping[key].name))
    trainTZ.append(str(tweetToTextMapping[key].timezone))
    trainUtc.append(str(tweetToTextMapping[key].utcOffset))



#Delete tweets and run gc
del(tweetToTextMapping)
import gc
gc.collect()


#########Preprocessing of data
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
import pandas

#1.) Binarize > 3000 target classes into one hot encoding
print(str(len(set(trainLabels))) +" different places known") #Number of different classes

from sklearn.preprocessing import LabelEncoder
utcEncoder = LabelEncoder()
utcEncoder.fit(trainLabels)

classes = utcEncoder.transform(trainLabels)

#Map class int representation to
colnames = [None]*len(set(classes))
for i in range(len(classes)):
    colnames[classes[i]] =  trainLabels[i]


#2.) Tokenize texts
import string
def my_filter():
    f = string.punctuation
    f += '\t\n\r…”'
    return f


#User-Description
MAX_DESC_SEQUENCE_LENGTH=10 #Median is 6
descriptionTokenizer = Tokenizer(nb_words=100000, filters=my_filter()) #Keep only top-N words
descriptionTokenizer.fit_on_texts(trainDescription)
trainDescription = descriptionTokenizer.texts_to_sequences(trainDescription)
trainDescription = np.asarray(trainDescription) #Convert to ndArraytop
trainDescription = pad_sequences(trainDescription, maxlen=MAX_DESC_SEQUENCE_LENGTH)

#Link-Mentions
MAX_URL_SEQUENCE_LENGTH=25
linkTokenizer = Tokenizer(70, filters='\t\n\r', char_level=True)
linkTokenizer.fit_on_texts(trainLinks)
trainLinks = linkTokenizer.texts_to_sequences(trainLinks)
trainLinks = np.asarray(trainLinks) #Convert to ndArraytop
trainLinks = pad_sequences(trainLinks, maxlen=MAX_URL_SEQUENCE_LENGTH)

#Location
MAX_LOC_SEQUENCE_LENGTH=3
locationTokenizer = Tokenizer(nb_words=80000, filters=my_filter()) #Keep only top-N words
locationTokenizer.fit_on_texts(trainLocation)
trainLocation = locationTokenizer.texts_to_sequences(trainLocation)
trainLocation = np.asarray(trainLocation) #Convert to ndArraytop
trainLocation = pad_sequences(trainLocation, maxlen=MAX_LOC_SEQUENCE_LENGTH)

#Source
sourceEncoder = LabelEncoder()
trainSource.append("unknown") #Temporarily add unknown for unknown classes
sourceEncoder.fit(trainSource)
trainSource.remove("unknown")
trainSource = sourceEncoder.transform(trainSource)
trainSource = pandas.get_dummies(trainSource).as_matrix()

#Text
MAX_TEXT_SEQUENCE_LENGTH=10
textTokenizer = Tokenizer(nb_words=100000, filters=my_filter()) #Keep only top-N words
textTokenizer.fit_on_texts(trainTexts)
trainTexts = textTokenizer.texts_to_sequences(trainTexts)
trainTexts = np.asarray(trainTexts) #Convert to ndArraytop
trainTexts = pad_sequences(trainTexts, maxlen=MAX_TEXT_SEQUENCE_LENGTH)

#Username
MAX_NAME_SEQUENCE_LENGTH=3
nameTokenizer = Tokenizer(nb_words=20000, filters=my_filter()) #Keep only top-N words
nameTokenizer.fit_on_texts(trainUserName)
trainUserName = nameTokenizer.texts_to_sequences(trainUserName)
trainUserName = np.asarray(trainUserName) #Convert to ndArraytop
trainUserName = pad_sequences(trainUserName, maxlen=MAX_NAME_SEQUENCE_LENGTH)

#TimeZone
MAX_TZ_SEQUENCE_LENGTH=4
timeZoneTokenizer = Tokenizer(nb_words=300) #Keep only top-N words
timeZoneTokenizer.fit_on_texts(trainTZ)
trainTZ = timeZoneTokenizer.texts_to_sequences(trainTZ)
trainTZ = np.asarray(trainTZ) #Convert to ndArraytop
trainTZ = pad_sequences(trainTZ, maxlen=MAX_TZ_SEQUENCE_LENGTH)

#UTC
utcEncoder = LabelEncoder()
utcEncoder.fit(trainUtc)
trainUtc = utcEncoder.transform(trainUtc)
trainUtc = np_utils.to_categorical(trainUtc)


#####Save result of preprocessing
import pickle
#1.) Save relevant processing data
filehandler = open(b"data/w-nut-latest/binaries/processors.obj","wb")
pickle.dump((descriptionTokenizer, linkTokenizer, locationTokenizer, sourceEncoder, textTokenizer, nameTokenizer, timeZoneTokenizer, utcEncoder, placeMedian, classes, colnames), filehandler)
filehandler.close()

#Save important variables
filehandler = open(b"data/w-nut-latest/binaries/vars.obj","wb")
pickle.dump((MAX_DESC_SEQUENCE_LENGTH, MAX_URL_SEQUENCE_LENGTH, MAX_LOC_SEQUENCE_LENGTH, MAX_TEXT_SEQUENCE_LENGTH, MAX_NAME_SEQUENCE_LENGTH, MAX_TZ_SEQUENCE_LENGTH), filehandler)
filehandler.close()

#2.) Save converted training data
filehandler = open(b"data/w-nut-latest/binaries/data.obj","wb")
pickle.dump((trainDescription, trainLinks, trainLocation, trainSource, trainTexts, trainUserName, trainTZ, trainUtc), filehandler, protocol=4)
filehandler.close()

