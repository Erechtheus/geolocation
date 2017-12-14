#Load stuff:
import pickle
import numpy as np
import time
from keras.models import Sequential
from keras.layers import LSTM,  Dropout, InputLayer, Dense, BatchNormalization, SpatialDropout1D
from keras.layers.embeddings import Embedding
import math

binaryPath="/media/philippe/5f695998-f5a5-4389-a2d8-4cf3ffa1288a/data/w-nut-latest/binaries/"
modelPath="data/w-nut-latest/models/"

#Load preprocessed data...
file = open(binaryPath +"processors.obj",'rb')
descriptionTokenizer, domainEncoder, tldEncoder, locationTokenizer, sourceEncoder, textTokenizer, nameTokenizer, timeZoneTokenizer, utcEncoder, langEncoder, timeEncoder, placeMedian, classes, colnames = pickle.load(file)

file = open(binaryPath +"vars.obj",'rb')
MAX_DESC_SEQUENCE_LENGTH, MAX_LOC_SEQUENCE_LENGTH, MAX_TEXT_SEQUENCE_LENGTH, MAX_NAME_SEQUENCE_LENGTH, MAX_TZ_SEQUENCE_LENGTH = pickle.load(file)

file = open(binaryPath +"data.obj",'rb')
trainDescription,  trainLocation, trainDomain, trainTld, trainSource, trainTexts, trainUserName, trainTZ, trainUtc, trainUserLang, trainCreatedAt = pickle.load(file)


##################Train
# create the model
batch_size = 256
nb_epoch = 5
verbosity=1

descriptionEmbeddings = 100
locEmbeddings = 50
textEmbeddings = 100
nameEmbeddings = 100
tzEmbeddings = 50



####################
#1.) Description Model
descriptionBranch = Sequential()
descriptionBranch.add(Embedding(descriptionTokenizer.num_words,
                                descriptionEmbeddings,
                                input_length=MAX_DESC_SEQUENCE_LENGTH
                                ))
descriptionBranch.add(SpatialDropout1D(rate=0.2))
descriptionBranch.add(BatchNormalization())
descriptionBranch.add(Dropout(0.2))
descriptionBranch.add(LSTM(units=30))
descriptionBranch.add(BatchNormalization())
descriptionBranch.add(Dropout(0.2, name="description"))

descriptionBranch.add(Dense(len(set(classes)), activation='softmax'))
descriptionBranch.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
descriptionHistory = descriptionBranch.fit(trainDescription, classes,
                    epochs=nb_epoch, batch_size=batch_size,
                    verbose=verbosity
                    )
print("descriptionBranch finished after " +str(time.time() - start))
descriptionBranch.save(modelPath +'descriptionBranchNorm.h5')




#####################
#2a.) Link Model for Domain
categorial = np.zeros((len(trainDomain), len(domainEncoder.classes_)), dtype="bool")
for i in range(len(trainDomain)):
    categorial[i, trainDomain[i]] = True
trainDomain = categorial


domainBranch = Sequential()
domainBranch.add(InputLayer(input_shape=(trainDomain.shape[1],)))
domainBranch.add(Dense(int(math.log2(trainDomain.shape[1])), activation='relu'))
domainBranch.add(BatchNormalization())
domainBranch.add(Dropout(0.2, name="domainName"))

domainBranch.add(Dense(len(set(classes)), activation='softmax'))
domainBranch.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
sourceHistory = domainBranch.fit(trainDomain, classes,
                    epochs=nb_epoch, batch_size=batch_size,
                    verbose=verbosity
                    )
print("tldBranch finished after " +str(time.time() - start))
domainBranch.save(modelPath + 'domainBranch.h5')



#2b.) Link Model for TLD
categorial = np.zeros((len(trainTld), len(tldEncoder.classes_)), dtype="bool")
for i in range(len(trainTld)):
    categorial[i, trainTld[i]] = True
trainTld = categorial


tldBranch = Sequential()
tldBranch.add(InputLayer(input_shape=(trainTld.shape[1],)))
tldBranch.add(Dense(int(math.log2(trainTld.shape[1])), activation='relu'))
tldBranch.add(BatchNormalization())
tldBranch.add(Dropout(0.2, name="tld"))


tldBranch.add(Dense(len(set(classes)), activation='softmax'))
tldBranch.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
sourceHistory = tldBranch.fit(trainTld, classes,
                    epochs=nb_epoch, batch_size=batch_size,
                    verbose=verbosity
                    )
print("tldBranch finished after " +str(time.time() - start))
tldBranch.save(modelPath + 'tldBranch.h5')


#2c.) TODO Merged Model
linkModel = Sequential()
linkModel.add(InputLayer(input_shape=((trainDomain.shape[1]+trainTld.shape[1]),)))
linkModel.add(Dense(int(math.log2(trainDomain.shape[1]+trainTld.shape[1])), activation='relu'))
linkModel.add(BatchNormalization())
linkModel.add(Dropout(0.2, name="linkModel"))


linkModel.add(Dense(len(set(classes)), activation='softmax'))
linkModel.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
sourceHistory = linkModel.fit(np.concatenate((trainDomain, trainTld), axis=1), classes,
                    epochs=nb_epoch, batch_size=batch_size,
                    verbose=verbosity
                    )
print("linkModel finished after " +str(time.time() - start))
linkModel.save(modelPath +'linkModel.h5')


#####################
#3.) location Model
locationBranch = Sequential()
locationBranch.add(Embedding(locationTokenizer.num_words,
                             locEmbeddings,
                    input_length=MAX_LOC_SEQUENCE_LENGTH
                    ))
locationBranch.add(SpatialDropout1D(rate=0.2))
locationBranch.add(BatchNormalization())
locationBranch.add(Dropout(0.2))
locationBranch.add(LSTM(units=30))
locationBranch.add(BatchNormalization())
locationBranch.add(Dropout(0.2, name="location"))

locationBranch.add(Dense(len(set(classes)), activation='softmax'))
locationBranch.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
locationHistory = locationBranch.fit(trainLocation, classes,
                    epochs=nb_epoch, batch_size=batch_size,
                    verbose=verbosity
                    )
print("locationHistory finished after " +str(time.time() - start))
locationBranch.save(modelPath +'locationBranchNorm.h5')


#####################
#4.) Source Mode
categorial = np.zeros((len(trainSource), len(sourceEncoder.classes_)), dtype="bool")
for i in range(len(trainSource)):
    categorial[i, trainSource[i]] = True
trainSource = categorial


sourceBranch = Sequential()
sourceBranch.add(InputLayer(input_shape=(trainSource.shape[1],)))
sourceBranch.add(Dense(int(math.log2(trainSource.shape[1])), activation='relu'))
sourceBranch.add(BatchNormalization())
sourceBranch.add(Dropout(0.2, name="source"))


sourceBranch.add(Dense(len(set(classes)), activation='softmax'))
sourceBranch.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
sourceHistory = sourceBranch.fit(trainSource, classes,
                    epochs=nb_epoch, batch_size=batch_size,
                    verbose=verbosity
                    )
print("sourceBranch finished after " +str(time.time() - start))
sourceBranch.save(modelPath +'sourceBranch.h5')



#####################
#5.) Text Model
textBranch = Sequential()
textBranch.add(Embedding(textTokenizer.num_words,
                         textEmbeddings ,
                        input_length=MAX_TEXT_SEQUENCE_LENGTH
                         ))
textBranch.add(SpatialDropout1D(rate=0.2))
textBranch.add(BatchNormalization())
textBranch.add(Dropout(0.2))
textBranch.add(LSTM(units=30))
textBranch.add(BatchNormalization())
textBranch.add(Dropout(0.2, name="text"))

textBranch.add(Dense(len(set(classes)), activation='softmax'))
textBranch.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
textHistory = textBranch.fit(trainTexts, classes,
                    epochs=nb_epoch, batch_size=batch_size,
                    verbose=verbosity
                    )
print("textBranch finished after " +str(time.time() - start))
textBranch.save(modelPath +'textBranchNorm.h5')



#####################
# 6.) Name Model
nameBranch = Sequential()
nameBranch.add(Embedding(nameTokenizer.num_words,
                         nameEmbeddings,
                         input_length=MAX_NAME_SEQUENCE_LENGTH
                         ))
nameBranch.add(SpatialDropout1D(rate=0.2))
nameBranch.add(BatchNormalization())
nameBranch.add(Dropout(0.2))
nameBranch.add(LSTM(units=30))
nameBranch.add(BatchNormalization())
nameBranch.add(Dropout(0.2, name="username"))

nameBranch.add(Dense(len(set(classes)), activation='softmax'))
nameBranch.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
nameHistory = nameBranch.fit(trainUserName, classes,
                    epochs=nb_epoch, batch_size=batch_size,
                    verbose=verbosity
                    )
print("nameBranch finished after " +str(time.time() - start))
nameBranch.save(modelPath +'nameBranchNorm.h5')





#####################
# 7.) TimeZone Model
tzBranch = Sequential()
tzBranch.add(Embedding(timeZoneTokenizer.num_words,
                       tzEmbeddings,
                       input_length=MAX_TZ_SEQUENCE_LENGTH
                       ))
tzBranch.add(SpatialDropout1D(rate=0.2))
tzBranch.add(BatchNormalization())
tzBranch.add(Dropout(0.2))
tzBranch.add(LSTM(units=30))
tzBranch.add(BatchNormalization())
tzBranch.add(Dropout(0.2, name="timezone"))

tzBranch.add(Dense(len(set(classes)), activation='softmax'))
tzBranch.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
tzHistory = tzBranch.fit(trainTZ, classes,
                    epochs=nb_epoch, batch_size=batch_size,
                    verbose=verbosity
                    )
print("tzBranch finished after " +str(time.time() - start))
tzBranch.save(modelPath +'tzBranchNorm.h5')



#####################
# 8.) UTC Model
categorial = np.zeros((len(trainUtc), len(utcEncoder.classes_)), dtype="bool")
for i in range(len(trainUtc)):
    categorial[i, trainUtc[i]] = True
trainUtc = categorial


utcBranch = Sequential()
utcBranch.add(InputLayer(input_shape=(trainUtc.shape[1],)))
utcBranch.add(Dense(int(math.log2(trainUtc.shape[1])), activation='relu'))
utcBranch.add(BatchNormalization())
utcBranch.add(Dropout(0.2, name="utc"))


utcBranch.add(Dense(len(set(classes)), activation='softmax'))
utcBranch.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
utcHistory = utcBranch.fit(trainUtc, classes,
                    epochs=nb_epoch, batch_size=batch_size,
                    verbose=verbosity
                    )
print("utcBranch finished after " +str(time.time() - start))
utcBranch.save(modelPath +'utcBranch.h5')


#9) "User Language
categorial = np.zeros((len(trainUserLang), len(langEncoder.classes_)), dtype="bool")
for i in range(len(trainUserLang)):
    categorial[i, trainUserLang[i]] = True
trainUserLang = categorial

userLangBranch = Sequential()
userLangBranch.add(InputLayer(input_shape=(trainUserLang.shape[1],)))
userLangBranch.add(Dense(int(math.log2(trainUserLang.shape[1])), activation='relu'))
userLangBranch.add(BatchNormalization())
userLangBranch.add(Dropout(0.2, name="userLang"))


userLangBranch.add(Dense(len(set(classes)), activation='softmax'))
userLangBranch.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
userLangHistory = userLangBranch.fit(trainUserLang, classes,
                    epochs=nb_epoch, batch_size=batch_size,
                    verbose=verbosity
                    )
print("userLangBranch finished after " +str(time.time() - start))
userLangBranch.save(modelPath +'userLangBranch.h5')


#10) #Tweet-Time (120)
categorial = np.zeros((len(trainCreatedAt), len(timeEncoder.classes_)), dtype="bool")
for i in range(len(trainCreatedAt)):
    categorial[i, trainCreatedAt[i]] = True
trainCreatedAt = categorial

tweetTimeBranch = Sequential()
tweetTimeBranch.add(InputLayer(input_shape=(trainCreatedAt.shape[1],)))
tweetTimeBranch.add(Dense(int(math.log2(trainCreatedAt.shape[1])), activation='relu'))
tweetTimeBranch.add(BatchNormalization())
tweetTimeBranch.add(Dropout(0.2, name="tweetTime"))


tweetTimeBranch.add(Dense(len(set(classes)), activation='softmax'))
tweetTimeBranch.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
userLangHistory = tweetTimeBranch.fit(trainCreatedAt, classes,
                    epochs=nb_epoch, batch_size=batch_size,
                    verbose=verbosity
                    )
print("tweetTimeBranch finished after " +str(time.time() - start))
tweetTimeBranch.save(modelPath +'tweetTimeBranch.h5')



#11) Merged sequential model
trainData = np.concatenate((trainDomain, trainTld, trainSource, trainUtc, trainUserLang, trainCreatedAt), axis=1)

categorialModel = Sequential()
categorialModel.add(InputLayer(input_shape=(trainData.shape[1],)))
categorialModel.add(Dense(int(math.log2(trainData.shape[1])), activation='relu'))
categorialModel.add(BatchNormalization())
categorialModel.add(Dropout(0.2, name="categorialModel"))


categorialModel.add(Dense(len(set(classes)), activation='softmax'))
categorialModel.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()

categorialModelHistory = categorialModel.fit(trainData, classes,
                    epochs=nb_epoch, batch_size=batch_size,
                    verbose=verbosity
                    )
print("categorialModel finished after " +str(time.time() - start))
categorialModel.save(modelPath +'categorialModel.h5')

