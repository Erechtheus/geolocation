#Load stuff:
#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

import time
from keras.layers import Dropout, Dense, BatchNormalization, SpatialDropout1D, LSTM, concatenate, Concatenate
from keras.layers.embeddings import Embedding
import math
import datetime
from keras import Input
from keras import Model

from keras.preprocessing.sequence import pad_sequences
import numpy as np
import gzip
import json

from representation import parseJsonLine, Place, extractPreprocessUrl
import pickle

#Rounds minutes to 15 Minutes ranges
def roundMinutes(x, base=15):
    return int(base * round(float(x)/base))

binaryPath= 'data/binaries/'    #Place where the serialized training data is
modelPath= 'data/models/'       #Place to store the models

#Load preprocessed data...
file = open(binaryPath +"processors.obj",'rb')
descriptionTokenizer, domainEncoder, tldEncoder, locationTokenizer, sourceEncoder, textTokenizer, nameTokenizer, timeZoneTokenizer, utcEncoder, langEncoder, timeEncoder, placeMedian, classes, colnames, classEncoder = pickle.load(file)

file = open(binaryPath +"vars.obj",'rb')
MAX_DESC_SEQUENCE_LENGTH, MAX_LOC_SEQUENCE_LENGTH, MAX_TEXT_SEQUENCE_LENGTH, MAX_NAME_SEQUENCE_LENGTH, MAX_TZ_SEQUENCE_LENGTH = pickle.load(file)


##################Train
# create the model
batch_size = 256
nb_epoch = 5
verbosity=2

descriptionEmbeddings = 100
locEmbeddings = 50
textEmbeddings = 100
nameEmbeddings = 100
tzEmbeddings = 50


trainingFile="data/train/training.twitter.json.gz" #File with  all ~9 Million training tweets
placesFile='data/train/training.json.gz'           #Place annotation provided by task organisers

#Parse and add gold-label for tweets
idToGold = {}
with gzip.open(placesFile,'rb') as file:
    for line in file:
        parsed_json = json.loads(line.decode('utf-8'))
        tweetId=int(parsed_json["tweet_id"])
        place = Place(name=parsed_json["tweet_city"], lat=parsed_json["tweet_latitude"], lon=parsed_json["tweet_longitude"])
        idToGold[tweetId] = place

numberOfTrainingsamples=9127900 #TODO zcat training.twitter.json.gz  | wc -l

def batch_generator(twitterFile, goldstandard, batch_size=64):
    while True: #TODO: needed?
        with gzip.open(twitterFile, 'rb') as file:
            trainDescriptions = []; trainLinks = []; trainLocation = []; trainSource = []; trainTexts = [];  trainUserName = []; trainTZ = []; trainUtc = []; trainUserLang = []; trainCreatedAt = [];  trainUserMentions = []; trainLabels = []
            for line in file:

                if len(trainDescriptions) == batch_size:
                    trainDescriptions = []; trainLinks = []; trainLocation = []; trainSource = []; trainTexts = []; trainUserName = []; trainTZ = []; trainUtc = []; trainUserLang = []; trainCreatedAt = []; trainUserMentions = []; trainLabels = []

                instance = parseJsonLine(line.decode('utf-8'))

                trainDescriptions.append(str(instance.description))
                trainLinks.append(extractPreprocessUrl(instance.urls))
                trainLocation.append(str(instance.location))
                trainSource.append(str(instance.source))
                trainTexts.append(instance.text)
                trainUserName.append(str(instance.name))
                trainTZ.append(str(instance.timezone))
                trainUtc.append(str(instance.utcOffset))
                trainUserLang.append(str(instance.userLanguage))
                trainCreatedAt.append(str(instance.createdAt.hour) + "-" + str(roundMinutes(instance.createdAt.minute)))
                trainUserMentions.append(instance.userMentions)

                trainLabel = goldstandard[instance.id]._name
                trainLabels.append(trainLabel)

                #print(str(instance.id) +"\t" +str(len(trainDescriptions)))

                if len(trainDescriptions) == batch_size:

                    #Descriptions
                    trainDescriptions = descriptionTokenizer.texts_to_sequences(trainDescriptions)
                    trainDescriptions = np.asarray(trainDescriptions)  # Convert to ndArraytop
                    trainDescriptions = pad_sequences(trainDescriptions, maxlen=MAX_DESC_SEQUENCE_LENGTH)

                    # Link-Mentions
                    trainDomain = list(map(lambda x: x[0], trainLinks))  # URL-Domain
                    categorial = np.zeros((len(trainDomain), len(domainEncoder.classes_)), dtype="bool")
                    for i in range(len(trainDomain)):
                        if trainDomain[i] in domainEncoder.classes_:
                            categorial[i, domainEncoder.transform([trainDomain[i]])[0]] = True
                    trainDomain = categorial

                    trainTld = list(map(lambda x: x[1], trainLinks))  # Url suffix; top level domain
                    categorial = np.zeros((len(trainTld), len(tldEncoder.classes_)), dtype="bool")
                    for i in range(len(trainTld)):
                        if trainTld[i] in tldEncoder.classes_:
                            categorial[i, tldEncoder.transform([trainTld[i]])[0]] = True
                    trainTld = categorial

                    # Location
                    trainLocation = locationTokenizer.texts_to_sequences(trainLocation)
                    trainLocation = np.asarray(trainLocation)  # Convert to ndArraytop
                    trainLocation = pad_sequences(trainLocation, maxlen=MAX_LOC_SEQUENCE_LENGTH)

                    # Source
                    trainSource = sourceEncoder.transform(trainSource)
                    categorial = np.zeros((len(trainSource), len(sourceEncoder.classes_)), dtype="bool")
                    for i in range(len(trainSource)):
                        categorial[i, trainSource[i]] = True
                    trainSource = categorial

                    #Text Tweet
                    trainTexts = textTokenizer.texts_to_sequences(trainTexts)
                    trainTexts = np.asarray(trainTexts)  # Convert to ndArraytop
                    trainTexts = pad_sequences(trainTexts, maxlen=MAX_TEXT_SEQUENCE_LENGTH)

                    #User Name
                    trainUserName = nameTokenizer.texts_to_sequences(trainUserName)
                    trainUserName = np.asarray(trainUserName)  # Convert to ndArraytop
                    trainUserName = pad_sequences(trainUserName, maxlen=MAX_NAME_SEQUENCE_LENGTH)

                    #Time Zone
                    trainTZ = timeZoneTokenizer.texts_to_sequences(trainTZ)
                    trainTZ = np.asarray(trainTZ)  # Convert to ndArraytop
                    trainTZ = pad_sequences(trainTZ, maxlen=MAX_TZ_SEQUENCE_LENGTH)

                    # UTC
                    trainUtc = utcEncoder.transform(trainUtc)
                    categorial = np.zeros((len(trainUtc), len(utcEncoder.classes_)), dtype="bool")
                    for i in range(len(trainUtc)):
                        categorial[i, trainUtc[i]] = True
                    trainUtc = categorial

                    # User-Language (63 languages)
                    trainUserLang = langEncoder.transform(trainUserLang)
                    categorial = np.zeros((len(trainUserLang), len(langEncoder.classes_)), dtype="bool")
                    for i in range(len(trainUserLang)):
                        categorial[i, trainUserLang[i]] = True
                    trainUserLang = categorial

                    # Tweet-Time (120 steps)
                    trainCreatedAt = timeEncoder.transform(trainCreatedAt)
                    categorial = np.zeros((len(trainCreatedAt), len(timeEncoder.classes_)), dtype="bool")
                    for i in range(len(trainCreatedAt)):
                        categorial[i, trainCreatedAt[i]] = True
                    trainCreatedAt = categorial

                    # class label
                    classes = classEncoder.transform(trainLabels)

                    #yield trainDescriptions, classes
                    yield ({'inputDescription': trainDescriptions,
                            'inputDomain': trainDomain,
                            'inputTld':trainTld,
                            'inputLocation': trainLocation,
                            'inputSource': trainSource,
                            'inputText' : trainTexts,
                            'inputUser' : trainUserName,
                            'inputTimeZone' : trainTZ,
                            'inputUTC' :trainUtc,
                            'inputUserLang' : trainUserLang,
                            'inputTweetTime': trainCreatedAt
                            },
                           #{'output': y}
                           classes
                           )


#1.) Description Model
descriptionBranchI = Input(shape=(None,), name="inputDescription")
descriptionBranch = Embedding(descriptionTokenizer.num_words,
                                descriptionEmbeddings,
                                input_length=MAX_DESC_SEQUENCE_LENGTH,
                                mask_zero=True
                                )(descriptionBranchI)
descriptionBranch = SpatialDropout1D(rate=0.2)(descriptionBranch)
descriptionBranch = BatchNormalization()(descriptionBranch)
descriptionBranch = Dropout(0.2)(descriptionBranch)
descriptionBranch = LSTM(units=30)(descriptionBranch)
descriptionBranch = BatchNormalization()(descriptionBranch)
descriptionBranch = Dropout(0.2, name="description")(descriptionBranch)
descriptionBranchO = Dense(len(set(classes)), activation='softmax')(descriptionBranch)

descriptionModel = Model(inputs=descriptionBranchI, outputs=descriptionBranchO)
descriptionModel.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
descriptionHistory = descriptionModel.fit_generator(generator=batch_generator(twitterFile=trainingFile, goldstandard=idToGold, batch_size=batch_size),
                    epochs=nb_epoch, steps_per_epoch=math.ceil(numberOfTrainingsamples/batch_size),
                    verbose=verbosity
                    )
print("descriptionBranch finished after " +str(datetime.timedelta(seconds=round(time.time() - start))))
descriptionModel.save(modelPath +'descriptionBranchNorm.h5')

#2a.) Link Model for Domain
domainBranchI = Input(shape=(len(domainEncoder.classes_),), name="inputDomain")
domainBranch = Dense(int(math.log2(len(domainEncoder.classes_))), input_shape=(len(domainEncoder.classes_),), activation='relu')(domainBranchI)
domainBranch = BatchNormalization()(domainBranch)
domainBranch = Dropout(0.2, name="domainName")(domainBranch)
domainBranchO = Dense(len(set(classes)), activation='softmax')(domainBranch)

domainModel = Model(inputs=domainBranchI, outputs=domainBranchO)
domainModel.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
sourceHistory = domainModel.fit_generator(generator=batch_generator(twitterFile=trainingFile, goldstandard=idToGold, batch_size=batch_size),
                    epochs=nb_epoch, steps_per_epoch=math.ceil(numberOfTrainingsamples/batch_size),
                    verbose=verbosity
                    )
print("tldBranch finished after " +str(datetime.timedelta(seconds=round(time.time() - start))))
domainModel.save(modelPath + 'domainBranch.h5')



#2b.) Link Model for TLD
tldBranchI = Input(shape=(len(tldEncoder.classes_),), name="inputTld")
tldBranch = Dense(int(math.log2(len(tldEncoder.classes_))), input_shape=(len(tldEncoder.classes_),), activation='relu')(tldBranchI)
tldBranch = BatchNormalization()(tldBranch)
tldBranch = Dropout(0.2, name="tld")(tldBranch)
tldBranchO = Dense(len(set(classes)), activation='softmax')(tldBranch)

tldBranchModel = Model(inputs=tldBranchI, outputs=tldBranchO)
tldBranchModel.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
sourceHistory = tldBranchModel.fit_generator(generator=batch_generator(twitterFile=trainingFile, goldstandard=idToGold, batch_size=batch_size),
                    epochs=nb_epoch, steps_per_epoch=math.ceil(numberOfTrainingsamples/batch_size),
                    verbose=verbosity
                    )
print("tldBranch finished after " +str(datetime.timedelta(seconds=round(time.time() - start))))
tldBranchModel.save(modelPath + 'tldBranch.h5')


#2c.)Merged Model
linkBranchI = concatenate([domainBranchI, tldBranchI])
linkBranch = Dense(int(math.log2(len(domainEncoder.classes_) + len(tldEncoder.classes_))), input_shape=((len(domainEncoder.classes_) + len(tldEncoder.classes_)),), activation='relu')(linkBranchI)
linkBranch = BatchNormalization()(linkBranch)
linkBranch = Dropout(0.2, name="linkModel")(linkBranch)
linkBranchO = Dense(len(set(classes)), activation='softmax')(linkBranch)

linkModel = Model(inputs=[domainBranchI, tldBranchI], outputs=linkBranchO)
linkModel.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
sourceHistory = linkModel.fit_generator(generator=batch_generator(twitterFile=trainingFile, goldstandard=idToGold, batch_size=batch_size),
                    epochs=nb_epoch, steps_per_epoch=math.ceil(numberOfTrainingsamples/batch_size),
                    verbose=verbosity
                    )
print("linkModel finished after " +str(datetime.timedelta(seconds=round(time.time() - start))))
linkModel.save(modelPath + 'linkModel.h5')

#####################
#3.) location Model
locationBranchI = Input(shape=(None,), name="inputLocation")
locationBranch = Embedding(locationTokenizer.num_words,
                             locEmbeddings,
                    input_length=MAX_LOC_SEQUENCE_LENGTH,
                    mask_zero=True
                    )(locationBranchI)
locationBranch = SpatialDropout1D(rate=0.2)(locationBranch)
locationBranch = BatchNormalization()(locationBranch)
locationBranch = Dropout(0.2)(locationBranch)
locationBranch = LSTM(units=30)(locationBranch)
locationBranch = BatchNormalization()(locationBranch)
locationBranch = Dropout(0.2, name="location")(locationBranch)
locationBranchO = Dense(len(set(classes)), activation='softmax')(locationBranch)

locationModel = Model(inputs=locationBranchI, outputs=locationBranchO)
locationModel.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
locationHistory = locationModel.fit_generator(generator=batch_generator(twitterFile=trainingFile, goldstandard=idToGold, batch_size=batch_size),
                    epochs=nb_epoch, steps_per_epoch=math.ceil(numberOfTrainingsamples/batch_size),
                    verbose=verbosity
                    )
print("locationHistory finished after " +str(datetime.timedelta(seconds=round(time.time() - start))))
locationModel.save(modelPath +'locationBranchNorm.h5')



#####################
#4.) Source Mode
sourceBranchI = Input(shape=(len(sourceEncoder.classes_),), name="inputSource")
sourceBranch = Dense(int(math.log2(len(sourceEncoder.classes_))), input_shape=(len(sourceEncoder.classes_),), activation='relu')(sourceBranchI)
sourceBranch = BatchNormalization()(sourceBranch)
sourceBranch = Dropout(0.2, name="source")(sourceBranch)
sourceBranchO = Dense(len(set(classes)), activation='softmax')(sourceBranch)

sourceModel = Model(inputs=sourceBranchI, outputs=sourceBranchO)
sourceModel.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
sourceHistory = sourceModel.fit_generator(generator=batch_generator(twitterFile=trainingFile, goldstandard=idToGold, batch_size=batch_size),
                    epochs=nb_epoch, steps_per_epoch=math.ceil(numberOfTrainingsamples/batch_size),
                    verbose=verbosity
                    )
print("sourceBranch finished after " +str(datetime.timedelta(seconds=round(time.time() - start))))
sourceModel.save(modelPath +'sourceBranch.h5')



#####################
#5.) Text Model
textBranchI = Input(shape=(None,), name="inputText")
textBranch = Embedding(textTokenizer.num_words,
                         textEmbeddings,
                        input_length=MAX_TEXT_SEQUENCE_LENGTH,
                        mask_zero=True
                         )(textBranchI)
textBranch = SpatialDropout1D(rate=0.2)(textBranch)
textBranch = BatchNormalization()(textBranch)
textBranch = Dropout(0.2)(textBranch)
textBranch = LSTM(units=30)(textBranch)
textBranch = BatchNormalization()(textBranch)
textBranch = Dropout(0.2, name="text")(textBranch)
textBranchO = Dense(len(set(classes)), activation='softmax')(textBranch)

textModel = Model(inputs=textBranchI, outputs=textBranchO)
textModel.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
textHistory = textModel.fit_generator(generator=batch_generator(twitterFile=trainingFile, goldstandard=idToGold, batch_size=batch_size),
                    epochs=nb_epoch, steps_per_epoch=math.ceil(numberOfTrainingsamples/batch_size),
                    verbose=verbosity
                    )
print("textBranch finished after " +str(datetime.timedelta(seconds=round(time.time() - start))))
textModel.save(modelPath +'textBranchNorm.h5')



#####################
# 6.) Name Model
nameBranchI = Input(shape=(None,), name="inputUser")
nameBranch = Embedding(nameTokenizer.num_words,
                         nameEmbeddings,
                         input_length=MAX_NAME_SEQUENCE_LENGTH,
                        mask_zero=True
                         )(nameBranchI)
nameBranch = SpatialDropout1D(rate=0.2)(nameBranch)
nameBranch = BatchNormalization()(nameBranch)
nameBranch = Dropout(0.2)(nameBranch)
nameBranch = LSTM(units=30)(nameBranch)
nameBranch = BatchNormalization()(nameBranch)
nameBranch = Dropout(0.2, name="username")(nameBranch)
nameBranchO = Dense(len(set(classes)), activation='softmax')(nameBranch)

nameModel = Model(inputs=nameBranchI, outputs=nameBranchO)
nameModel.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
nameHistory = nameModel.fit_generator(generator=batch_generator(twitterFile=trainingFile, goldstandard=idToGold, batch_size=batch_size),
                    epochs=nb_epoch, steps_per_epoch=math.ceil(numberOfTrainingsamples/batch_size),
                    verbose=verbosity
                    )
print("nameBranch finished after " +str(datetime.timedelta(seconds=round(time.time() - start))))
nameModel.save(modelPath +'nameBranchNorm.h5')


#####################
# 7.) TimeZone Model
tzBranchI = Input(shape=(None,), name="inputTimeZone")
tzBranch = Embedding(timeZoneTokenizer.num_words,
                       tzEmbeddings,
                       input_length=MAX_TZ_SEQUENCE_LENGTH,
                        mask_zero=True
                       )(tzBranchI)
tzBranch = SpatialDropout1D(rate=0.2)(tzBranch)
tzBranch = BatchNormalization()(tzBranch)
tzBranch = Dropout(0.2)(tzBranch)
tzBranch = LSTM(units=30)(tzBranch)
tzBranch = BatchNormalization()(tzBranch)
tzBranch = Dropout(0.2, name="timezone")(tzBranch)
tzBranchO = Dense(len(set(classes)), activation='softmax')(tzBranch)

tzBranchModel = Model(inputs=tzBranchI, outputs=tzBranchO)
tzBranchModel.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
tzHistory = tzBranchModel.fit_generator(generator=batch_generator(twitterFile=trainingFile, goldstandard=idToGold, batch_size=batch_size),
                    epochs=nb_epoch, steps_per_epoch=math.ceil(numberOfTrainingsamples/batch_size),
                    verbose=verbosity
                    )
print("tzBranch finished after " +str(datetime.timedelta(seconds=round(time.time() - start))))
tzBranchModel.save(modelPath +'tzBranchNorm.h5')



#####################
# 8.) UTC Model
utcBranchI = Input(shape=(len(utcEncoder.classes_),), name="inputUTC")
utcBranch =  Dense(int(math.log2(len(utcEncoder.classes_))), activation='relu')(utcBranchI)
utcBranch = BatchNormalization()(utcBranch)
utcBranch = Dropout(0.2, name="utc")(utcBranch)
utcBranchO = Dense(len(set(classes)), activation='softmax')(utcBranch)

utcBranchModel = Model(inputs=utcBranchI, outputs=utcBranchO)
utcBranchModel.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
utcHistory = utcBranchModel.fit_generator(generator=batch_generator(twitterFile=trainingFile, goldstandard=idToGold, batch_size=batch_size),
                    epochs=nb_epoch, steps_per_epoch=math.ceil(numberOfTrainingsamples/batch_size),
                    verbose=verbosity
                    )
print("utcBranch finished after " +str(datetime.timedelta(seconds=round(time.time() - start))))
utcBranchModel.save(modelPath +'utcBranch.h5')


#9) "User Language
userLangBranchI = Input(shape=( len(langEncoder.classes_),), name="inputUserLang")
userLangBranch = Dense(int(math.log2( len(langEncoder.classes_))),input_shape=( len(langEncoder.classes_),), activation='relu')(userLangBranchI)
userLangBranch = BatchNormalization()(userLangBranch)
userLangBranch = Dropout(0.2, name="userLang")(userLangBranch)
userLangBranchO = Dense(len(set(classes)), activation='softmax')(userLangBranch)

userLangModel = Model(inputs=userLangBranchI, outputs=userLangBranchO)
userLangModel.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
userLangHistory = userLangModel.fit_generator(generator=batch_generator(twitterFile=trainingFile, goldstandard=idToGold, batch_size=batch_size),
                    epochs=nb_epoch, steps_per_epoch=math.ceil(numberOfTrainingsamples/batch_size),
                    verbose=verbosity
                    )
print("userLangBranch finished after " +str(datetime.timedelta(seconds=round(time.time() - start))))
userLangModel.save(modelPath +'userLangBranch.h5')


#10) #Tweet-Time (120)
tweetTimeBranchI = Input(shape=(len(timeEncoder.classes_),), name="inputTweetTime")
tweetTimeBranch = Dense(int(math.log2(len(timeEncoder.classes_))), input_shape=(len(timeEncoder.classes_),), activation='relu')(tweetTimeBranchI)
tweetTimeBranch = BatchNormalization()(tweetTimeBranch)
tweetTimeBranch = Dropout(0.2, name="tweetTime")(tweetTimeBranch)
tweetTimeBranchO = Dense(len(set(classes)), activation='softmax')(tweetTimeBranch)

tweetTimeModel = Model(inputs=tweetTimeBranchI, outputs=tweetTimeBranchO)
tweetTimeModel.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
userLangHistory = tweetTimeModel.fit_generator(generator=batch_generator(twitterFile=trainingFile, goldstandard=idToGold, batch_size=batch_size),
                    epochs=nb_epoch, steps_per_epoch=math.ceil(numberOfTrainingsamples/batch_size),
                    verbose=verbosity
                    )
print("tweetTimeBranch finished after " +str(datetime.timedelta(seconds=round(time.time() - start))))
tweetTimeModel.save(modelPath +'tweetTimeBranch.h5')



#11) Merged sequential model
"""
trainData = np.concatenate((trainDomain, trainTld, trainSource, trainUserLang, trainCreatedAt), axis=1)

categorialBranchI = Input(shape=(trainData.shape[1],), name="inputCategorial")
categorialBranch = Dense(int(math.log2(trainData.shape[1])), input_shape=(trainData.shape[1],), activation='relu')(categorialBranchI)
categorialBranch = BatchNormalization()(categorialBranch)
categorialBranch = Dropout(0.2, name="categorialModel")(categorialBranch)
categorialBranchO = Dense(len(set(classes)), activation='softmax')(categorialBranch)


categorialModel = Model(inputs=categorialBranchI, outputs=categorialBranchO)
categorialModel.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()

categorialModelHistory = categorialModel.fit(trainData, classes,
                                              epochs=nb_epoch, batch_size=batch_size,
                                              verbose=verbosity
                                              )
print("categorialModel finished after " +str(datetime.timedelta(time.time() - start)))
categorialModel.save(modelPath + 'categorialModel.h5')
"""