##Performs preprocessing for twitter-training data
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import numpy as np
import gzip
import json
from representation import parseJsonLine, Place, extractPreprocessUrl
from collections import Counter
import pickle
import gc
import string


trainingFile="data/train/training.twitter.json.gz" #File with  all ~9 Million training tweets
placesFile='data/train/training.json.gz'           #Place annotation provided by task organisers
binaryPath= 'data/binaries/'            #Place to store the results

#Parser Twitter-JSON; loads all tweets into RAM (not very memory efficient!)
tweetToTextMapping= {} # Map<Twitter-ID; tweet>
with gzip.open(trainingFile,'rb') as file:
    for line in file:
        instance = parseJsonLine(line.decode('utf-8'))
        tweetToTextMapping[instance.id] = instance

#Parse and add gold-label for tweets
with gzip.open(placesFile,'rb') as file:
    for line in file:
        parsed_json = json.loads(line.decode('utf-8'))
        tweetId=int(parsed_json["tweet_id"])
        if(tweetId in tweetToTextMapping):
            place = Place(name=parsed_json["tweet_city"], lat=parsed_json["tweet_latitude"], lon=parsed_json["tweet_longitude"])
            tweetToTextMapping[tweetId].place = place

print(str(len(tweetToTextMapping.keys())) + " tweets for training are found")


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
trainUserLang =[]
trainSinTime = []
trainCosTime = []
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
    trainUserLang.append(str(tweetToTextMapping[key].userLanguage))

    t = tweetToTextMapping[key].createdAt.hour * 60 * 60 + tweetToTextMapping[key].createdAt.minute * 60 + tweetToTextMapping[key].createdAt.second
    t = 2*np.pi*t/(24*60*60)
    trainSinTime.append(np.sin(t))
    trainCosTime.append(np.cos(t))

trainCreatedAt = np.column_stack((trainSinTime, trainCosTime))
del(trainSinTime)
del(trainCosTime)

#Delete tweets and run gc
del(tweetToTextMapping)
gc.collect()


#########Preprocessing of data
#1.) Binarize > 3000 target classes into one hot encoding
print(str(len(set(trainLabels))) +" different places known") #Number of different classes

classEncoder = LabelEncoder()
classEncoder.fit(trainLabels)
classes = classEncoder.transform(trainLabels)

#Map class int representation to
colnames = [None]*len(set(classes))
for i in range(len(classes)):
    colnames[classes[i]] =  trainLabels[i]

#2.) Tokenize texts
def my_filter():
    f = string.punctuation
    f += '\t\n\r…”'
    return f

#User-Description
print("User Description")
descriptionTokenizer = Tokenizer(num_words=100000, filters=my_filter()) #Keep only top-N words
descriptionTokenizer.fit_on_texts(trainDescription)
trainDescription = descriptionTokenizer.texts_to_sequences(trainDescription)
trainDescription = np.asarray(trainDescription) #Convert to ndArraytop
trainDescription = pad_sequences(trainDescription)

#Link-Mentions
print("Links")
trainDomain = list(map(lambda x : x[0], trainLinks)) #URL-Domain

count = Counter(trainDomain)
for i in range(len(trainDomain)):
    if count[trainDomain[i]] < 50:
        trainDomain[i] = ''

domainEncoder = LabelEncoder()
domainEncoder.fit(trainDomain)
trainDomain = domainEncoder.transform(trainDomain)


trainTld = list(map(lambda x : x[1], trainLinks)) #Url suffix; top level domain

count = Counter(trainTld)
for i in range(len(trainTld)):
    if count[trainTld[i]] < 50:
        trainTld[i] = ''
tldEncoder = LabelEncoder()
tldEncoder.fit(trainTld)
trainTld = tldEncoder.transform(trainTld)

#Location
print("User Location")
locationTokenizer = Tokenizer(num_words=80000, filters=my_filter()) #Keep only top-N words
locationTokenizer.fit_on_texts(trainLocation)
trainLocation = locationTokenizer.texts_to_sequences(trainLocation)
trainLocation = np.asarray(trainLocation) #Convert to ndArraytop
trainLocation = pad_sequences(trainLocation)

#Source
print("Device source")
sourceEncoder = LabelEncoder()
sourceEncoder.fit(trainSource)
trainSource = sourceEncoder.transform(trainSource)

#Text
print("Tweet Text")
textTokenizer = Tokenizer(num_words=100000, filters=my_filter()) #Keep only top-N words
textTokenizer.fit_on_texts(trainTexts)
trainTexts = textTokenizer.texts_to_sequences(trainTexts)
trainTexts = np.asarray(trainTexts) #Convert to ndArraytop
trainTexts = pad_sequences(trainTexts)

#Username
print("Username")
nameTokenizer = Tokenizer(num_words=20000, filters=my_filter()) #Keep only top-N words
nameTokenizer.fit_on_texts(trainUserName)
trainUserName = nameTokenizer.texts_to_sequences(trainUserName)
trainUserName = np.asarray(trainUserName) #Convert to ndArraytop
trainUserName = pad_sequences(trainUserName)

#TimeZone
print("TimeZone")
timeZoneTokenizer = Tokenizer(num_words=300) #Keep only top-N words
timeZoneTokenizer.fit_on_texts(trainTZ)
trainTZ = timeZoneTokenizer.texts_to_sequences(trainTZ)
trainTZ = np.asarray(trainTZ) #Convert to ndArraytop
trainTZ = pad_sequences(trainTZ)

#UTC
print("UTC")
utcEncoder = LabelEncoder()
utcEncoder.fit(trainUtc)
trainUtc = utcEncoder.transform(trainUtc)

#User-Language (63 languages)
print("User Language")
langEncoder = LabelEncoder()
langEncoder.fit(trainUserLang)
trainUserLang = langEncoder.transform(trainUserLang)

#####Save result of preprocessing
#1.) Save relevant processing data
filehandler = open(binaryPath + "processors.obj", "wb")
pickle.dump((descriptionTokenizer, domainEncoder, tldEncoder, locationTokenizer, sourceEncoder, textTokenizer, nameTokenizer, timeZoneTokenizer, utcEncoder, langEncoder, placeMedian, colnames, classEncoder ), filehandler)
filehandler.close()

#2.) Save converted training data
filehandler = open(binaryPath + "data.obj", "wb")
pickle.dump((trainDescription, trainLocation, trainDomain, trainTld, trainSource, trainTexts, trainUserName, trainTZ, trainUtc, trainUserLang, trainCreatedAt, classes), filehandler, protocol=4)
filehandler.close()