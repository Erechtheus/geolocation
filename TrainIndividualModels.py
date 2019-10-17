#Load stuff:
from sklearn.utils import shuffle
import os
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pickle
import numpy as np
import time
from keras.layers import Dropout, Dense, BatchNormalization, SpatialDropout1D, LSTM
from keras.layers.embeddings import Embedding
import math
import datetime
from keras import Input
from keras import Model


binaryPath= 'data/binaries/'    #Place where the serialized training data is
modelPath= 'data/models/'       #Place to store the models

#Load preprocessed data...
file = open(binaryPath +"processors.obj",'rb')
descriptionTokenizer, domainEncoder, tldEncoder, locationTokenizer, sourceEncoder, textTokenizer, nameTokenizer, timeZoneTokenizer, utcEncoder, langEncoder, placeMedian, colnames, classEncoder  = pickle.load(file)

file = open(binaryPath +"data.obj",'rb')
trainDescription, trainLocation, trainDomain, trainTld, trainSource, trainTexts, trainUserName, trainTZ, trainUtc, trainUserLang, trainCreatedAt, classes= pickle.load(file)

#Shuffle train-data
trainDescription,  trainLocation, trainDomain, trainTld, trainSource, trainTexts, trainUserName, trainTZ, trainUtc, trainUserLang, trainCreatedAt, classes = shuffle(trainDescription,  trainLocation, trainDomain, trainTld, trainSource, trainTexts, trainUserName, trainTZ, trainUtc, trainUserLang, trainCreatedAt, classes, random_state=1202)

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
validation_split = 0.01 #91279 samples for validation

callbacks = [
#     EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=2, verbose=1, restore_best_weights=True),
#     ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_delta=1e-4, patience=2, cooldown=1, verbose=1),
##     ModelCheckpoint(filepath='twitter.h5', monitor='loss', verbose=0, save_best_only=True),
]


####################
#1.) Description Model
descriptionBranchI = Input(shape=(None,), name="inputDescription")
descriptionBranch = Embedding(descriptionTokenizer.num_words,
                                descriptionEmbeddings,
                                #input_length=MAX_DESC_SEQUENCE_LENGTH,
                                mask_zero=True
                                )(descriptionBranchI)
descriptionBranch = SpatialDropout1D(rate=0.2)(descriptionBranch) #Masks the same embedding element for all tokens
descriptionBranch = BatchNormalization()(descriptionBranch)
descriptionBranch = Dropout(0.2)(descriptionBranch)
descriptionBranch = LSTM(units=30)(descriptionBranch)
descriptionBranch = BatchNormalization()(descriptionBranch)
descriptionBranch = Dropout(0.2, name="description")(descriptionBranch)
descriptionBranchO = Dense(len(set(classes)), activation='softmax')(descriptionBranch)

descriptionModel = Model(inputs=descriptionBranchI, outputs=descriptionBranchO)
descriptionModel.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
descriptionHistory = descriptionModel.fit(trainDescription, classes,
                    epochs=nb_epoch, batch_size=batch_size,
                    verbose=verbosity, validation_split=validation_split,callbacks=callbacks
                    )
print("descriptionBranch finished after " +str(datetime.timedelta(seconds=round(time.time() - start))))
descriptionModel.save(modelPath +'descriptionBranchNorm.h5')




#####################
#2a.) Link Model for Domain
categorial = np.zeros((len(trainDomain), len(domainEncoder.classes_)), dtype="bool")
for i in range(len(trainDomain)):
    categorial[i, trainDomain[i]] = True
trainDomain = categorial

domainBranchI = Input(shape=(trainDomain.shape[1],), name="inputDomain")
domainBranch = Dense(int(math.log2(trainDomain.shape[1])), input_shape=(trainDomain.shape[1],), activation='relu')(domainBranchI)
domainBranch = BatchNormalization()(domainBranch)
domainBranch = Dropout(0.2, name="domainName")(domainBranch)
domainBranchO = Dense(len(set(classes)), activation='softmax')(domainBranch)

domainModel = Model(inputs=domainBranchI, outputs=domainBranchO)
domainModel.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
sourceHistory = domainModel.fit(trainDomain, classes,
                    epochs=nb_epoch, batch_size=batch_size,
                    verbose=verbosity, validation_split=validation_split,callbacks=callbacks
                    )
print("tldBranch finished after " +str(datetime.timedelta(seconds=round(time.time() - start))))
domainModel.save(modelPath + 'domainBranch.h5')



#2b.) Link Model for TLD
categorial = np.zeros((len(trainTld), len(tldEncoder.classes_)), dtype="bool")
for i in range(len(trainTld)):
    categorial[i, trainTld[i]] = True
trainTld = categorial


tldBranchI = Input(shape=(trainTld.shape[1],), name="inputTld")
tldBranch = Dense(int(math.log2(trainTld.shape[1])), input_shape=(trainTld.shape[1],), activation='relu')(tldBranchI)
tldBranch = BatchNormalization()(tldBranch)
tldBranch = Dropout(0.2, name="tld")(tldBranch)
tldBranchO = Dense(len(set(classes)), activation='softmax')(tldBranch)

tldBranchModel = Model(inputs=tldBranchI, outputs=tldBranchO)
tldBranchModel.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
sourceHistory = tldBranchModel.fit(trainTld, classes,
                    epochs=nb_epoch, batch_size=batch_size,
                    verbose=verbosity, validation_split=validation_split,callbacks=callbacks
                    )
print("tldBranch finished after " +str(datetime.timedelta(seconds=round(time.time() - start))))
tldBranchModel.save(modelPath + 'tldBranch.h5')


#2c.)Merged Model
linkBranchI= Input(shape=((trainDomain.shape[1] + trainTld.shape[1]),), name="inputLink")
linkBranch = Dense(int(math.log2(trainDomain.shape[1] + trainTld.shape[1])), input_shape=((trainDomain.shape[1] + trainTld.shape[1]),), activation='relu')(linkBranchI)
linkBranch = BatchNormalization()(linkBranch)
linkBranch = Dropout(0.2, name="linkModel")(linkBranch)
linkBranchO = Dense(len(set(classes)), activation='softmax')(linkBranch)

linkModel = Model(inputs=linkBranchI, outputs=linkBranchO)
linkModel.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
sourceHistory = linkModel.fit(np.concatenate((trainDomain, trainTld), axis=1), classes,
                               epochs=nb_epoch, batch_size=batch_size,
                               verbose=verbosity, validation_split=validation_split,callbacks=callbacks
                               )
print("linkModel finished after " +str(datetime.timedelta(seconds=round(time.time() - start))))
linkModel.save(modelPath + 'linkModel.h5')


#####################
#3.) location Model
locationBranchI = Input(shape=(None,), name="inputLocation")
locationBranch = Embedding(locationTokenizer.num_words,
                             locEmbeddings,
                    #input_length=MAX_LOC_SEQUENCE_LENGTH,
                    mask_zero=True
                    )(locationBranchI)
locationBranch = SpatialDropout1D(rate=0.2)(locationBranch)#Masks the same embedding element for all tokens
locationBranch = BatchNormalization()(locationBranch)
locationBranch = Dropout(0.2)(locationBranch)
locationBranch = LSTM(units=30)(locationBranch)
locationBranch = BatchNormalization()(locationBranch)
locationBranch = Dropout(0.2, name="location")(locationBranch)
locationBranchO = Dense(len(set(classes)), activation='softmax')(locationBranch)

locationModel = Model(inputs=locationBranchI, outputs=locationBranchO)
locationModel.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
locationHistory = locationModel.fit(trainLocation, classes,
                    epochs=nb_epoch, batch_size=batch_size,
                    verbose=verbosity, validation_split=validation_split,callbacks=callbacks
                    )
print("locationHistory finished after " +str(datetime.timedelta(seconds=round(time.time() - start))))
locationModel.save(modelPath +'locationBranchNorm.h5')


#####################
#4.) Source Mode
categorial = np.zeros((len(trainSource), len(sourceEncoder.classes_)), dtype="bool")
for i in range(len(trainSource)):
    categorial[i, trainSource[i]] = True
trainSource = categorial


sourceBranchI = Input(shape=(trainSource.shape[1],), name="inputSource")
sourceBranch = Dense(int(math.log2(trainSource.shape[1])), input_shape=(trainSource.shape[1],), activation='relu')(sourceBranchI)
sourceBranch = BatchNormalization()(sourceBranch)
sourceBranch = Dropout(0.2, name="source")(sourceBranch)
sourceBranchO = Dense(len(set(classes)), activation='softmax')(sourceBranch)

sourceModel = Model(inputs=sourceBranchI, outputs=sourceBranchO)
sourceModel.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
sourceHistory = sourceModel.fit(trainSource, classes,
                    epochs=nb_epoch, batch_size=batch_size,
                    verbose=verbosity, validation_split=validation_split,callbacks=callbacks
                    )
print("sourceBranch finished after " +str(datetime.timedelta(seconds=round(time.time() - start))))
sourceModel.save(modelPath +'sourceBranch.h5')



#####################
#5.) Text Model
textBranchI = Input(shape=(None,), name="inputText")
textBranch = Embedding(textTokenizer.num_words,
                         textEmbeddings,
                        #input_length=MAX_TEXT_SEQUENCE_LENGTH,
                        mask_zero=True
                         )(textBranchI)
textBranch = SpatialDropout1D(rate=0.2)(textBranch) #Masks the same embedding element for all tokens
textBranch = BatchNormalization()(textBranch)
textBranch = Dropout(0.2)(textBranch)
textBranch = LSTM(units=30)(textBranch)
textBranch = BatchNormalization()(textBranch)
textBranch = Dropout(0.2, name="text")(textBranch)
textBranchO = Dense(len(set(classes)), activation='softmax')(textBranch)

textModel = Model(inputs=textBranchI, outputs=textBranchO)
textModel.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
textHistory = textModel.fit(trainTexts, classes,
                    epochs=nb_epoch, batch_size=batch_size,
                    verbose=verbosity, validation_split=validation_split,callbacks=callbacks
                    )
print("textBranch finished after " +str(datetime.timedelta(seconds=round(time.time() - start))))
textModel.save(modelPath +'textBranchNorm.h5')



#####################
# 6.) Name Model
nameBranchI = Input(shape=(None,), name="inputName")
nameBranch = Embedding(nameTokenizer.num_words,
                         nameEmbeddings,
                         #input_length=MAX_NAME_SEQUENCE_LENGTH,
                        mask_zero=True
                         )(nameBranchI)
nameBranch = SpatialDropout1D(rate=0.2)(nameBranch) #Masks the same embedding element for all tokens
nameBranch = BatchNormalization()(nameBranch)
nameBranch = Dropout(0.2)(nameBranch)
nameBranch = LSTM(units=30)(nameBranch)
nameBranch = BatchNormalization()(nameBranch)
nameBranch = Dropout(0.2, name="username")(nameBranch)
nameBranchO = Dense(len(set(classes)), activation='softmax')(nameBranch)

nameModel = Model(inputs=nameBranchI, outputs=nameBranchO)
nameModel.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
nameHistory = nameModel.fit(trainUserName, classes,
                    epochs=nb_epoch, batch_size=batch_size,
                    verbose=verbosity, validation_split=validation_split,callbacks=callbacks
                    )
print("nameBranch finished after " +str(datetime.timedelta(seconds=round(time.time() - start))))
nameModel.save(modelPath +'nameBranchNorm.h5')


#####################
# 7.) TimeZone Model
tzBranchI = Input(shape=(None,), name="inputTimeZone")
tzBranch = Embedding(timeZoneTokenizer.num_words,
                       tzEmbeddings,
                       #input_length=MAX_TZ_SEQUENCE_LENGTH,
                        mask_zero=True
                       )(tzBranchI)
tzBranch = SpatialDropout1D(rate=0.2)(tzBranch) #Masks the same embedding element for all tokens
tzBranch = BatchNormalization()(tzBranch)
tzBranch = Dropout(0.2)(tzBranch)
tzBranch = LSTM(units=30)(tzBranch)
tzBranch = BatchNormalization()(tzBranch)
tzBranch = Dropout(0.2, name="timezone")(tzBranch)
tzBranchO = Dense(len(set(classes)), activation='softmax')(tzBranch)

tzBranchModel = Model(inputs=tzBranchI, outputs=tzBranchO)
tzBranchModel.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
tzHistory = tzBranchModel.fit(trainTZ, classes,
                    epochs=nb_epoch, batch_size=batch_size,
                    verbose=verbosity, validation_split=validation_split,callbacks=callbacks
                    )
print("tzBranch finished after " +str(datetime.timedelta(seconds=round(time.time() - start))))
tzBranchModel.save(modelPath +'tzBranchNorm.h5')



#####################
# 8.) UTC Model
categorial = np.zeros((len(trainUtc), len(utcEncoder.classes_)), dtype="bool")
for i in range(len(trainUtc)):
    categorial[i, trainUtc[i]] = True
trainUtc = categorial


utcBranchI = Input(shape=(trainUtc.shape[1],), name="inputUTC")
utcBranch =  Dense(int(math.log2(trainUtc.shape[1])), activation='relu')(utcBranchI)
utcBranch = BatchNormalization()(utcBranch)
utcBranch = Dropout(0.2, name="utc")(utcBranch)
utcBranchO = Dense(len(set(classes)), activation='softmax')(utcBranch)

utcBranchModel = Model(inputs=utcBranchI, outputs=utcBranchO)
utcBranchModel.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
utcHistory = utcBranchModel.fit(trainUtc, classes,
                    epochs=nb_epoch, batch_size=batch_size,
                    verbose=verbosity, validation_split=validation_split,callbacks=callbacks
                    )
print("utcBranch finished after " +str(datetime.timedelta(seconds=round(time.time() - start))))
utcBranchModel.save(modelPath +'utcBranch.h5')


#9) "User Language
categorial = np.zeros((len(trainUserLang), len(langEncoder.classes_)), dtype="bool")
for i in range(len(trainUserLang)):
    categorial[i, trainUserLang[i]] = True
trainUserLang = categorial

userLangBranchI = Input(shape=(trainUserLang.shape[1],), name="inputUserLang")
userLangBranch = Dense(int(math.log2(trainUserLang.shape[1])),input_shape=(trainUserLang.shape[1],), activation='relu')(userLangBranchI)
userLangBranch = BatchNormalization()(userLangBranch)
userLangBranch = Dropout(0.2, name="userLang")(userLangBranch)
userLangBranchO = Dense(len(set(classes)), activation='softmax')(userLangBranch)

userLangModel = Model(inputs=userLangBranchI, outputs=userLangBranchO)
userLangModel.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
userLangHistory = userLangModel.fit(trainUserLang, classes,
                    epochs=nb_epoch, batch_size=batch_size,
                    verbose=verbosity, validation_split=validation_split,callbacks=callbacks
                    )
print("userLangBranch finished after " +str(datetime.timedelta(seconds=round(time.time() - start))))
userLangModel.save(modelPath +'userLangBranch.h5')

#10a Tweet-time (as number)
tweetTimeBranchI = Input(shape=(trainCreatedAt.shape[1],), name="inputTweetTime")
tweetTimeBranch = Dense(2, name="tweetTime")(tweetTimeBranchI)# simple-no-operation layer, which is used in the merged model especially
tweetTimeBranchO = Dense(len(set(classes)), activation='softmax')(tweetTimeBranch)

tweetTimeModel = Model(inputs=tweetTimeBranchI, outputs=tweetTimeBranchO)
tweetTimeModel.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()

timeHistory = tweetTimeModel.fit(trainCreatedAt, classes,
                               epochs=nb_epoch, batch_size=batch_size,
                               verbose=verbosity, validation_split=validation_split,callbacks=callbacks
                               )
print("tweetTimeModel finished after " +str(datetime.timedelta(seconds=round(time.time() - start))))
tweetTimeModel.save(modelPath + 'tweetTimeBranch.h5')


#10) #Tweet-Time (120-categorial; instead of number)
"""
categorial = np.zeros((len(trainCreatedAt), len(timeEncoder.classes_)), dtype="bool")
for i in range(len(trainCreatedAt)):
    categorial[i, trainCreatedAt[i]] = True
trainCreatedAt = categorial

tweetTimeBranchI = Input(shape=(trainCreatedAt.shape[1],), name="inputTweetTime")
tweetTimeBranch = Dense(int(math.log2(trainCreatedAt.shape[1])), input_shape=(trainCreatedAt.shape[1],), activation='relu')(tweetTimeBranchI)
tweetTimeBranch = BatchNormalization()(tweetTimeBranch)
tweetTimeBranch = Dropout(0.2, name="tweetTime")(tweetTimeBranch)
tweetTimeBranchO = Dense(len(set(classes)), activation='softmax')(tweetTimeBranch)

tweetTimeModel = Model(inputs=tweetTimeBranchI, outputs=tweetTimeBranchO)
tweetTimeModel.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()

timeHistory = tweetTimeModel.fit(trainCreatedAt, classes,
                               epochs=nb_epoch, batch_size=batch_size,
                               verbose=verbosity, validation_split=validation_split,callbacks=callbacks
                               )
print("tweetTimeModel finished after " +str(datetime.timedelta(seconds=round(time.time() - start))))
tweetTimeModel.save(modelPath + 'tweetTimeBranch.h5')
"""


#11) Merged sequential model
trainData = np.concatenate((trainDomain, trainTld, trainSource, trainUserLang), axis=1)

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
                                              verbose=verbosity, validation_split=validation_split,callbacks=callbacks
                                              )
print("categorialModel finished after " +str(datetime.timedelta(time.time() - start)))
categorialModel.save(modelPath + 'categorialModel.h5')