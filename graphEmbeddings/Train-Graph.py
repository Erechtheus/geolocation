import pickle
import numpy as np
import time
from keras.layers import Dropout, Dense, BatchNormalization
import math
import datetime
from keras import Input
from keras import Model

binaryPath= 'data/binaries/'    #Place where the serialized training data is
modelPath= 'data/models/'       #Place to store the models


#Load preprocessed data...
file = open("/home/philippe/PycharmProjects/GEM/examples/data/embedding.obj", "rb")
embedding, idMapping =  pickle.load(file)

file = open(binaryPath +"processors.obj",'rb')
descriptionTokenizer, domainEncoder, tldEncoder, locationTokenizer, sourceEncoder, textTokenizer, nameTokenizer, timeZoneTokenizer, utcEncoder, langEncoder, timeEncoder, placeMedian, classes, colnames = pickle.load(file)

file = open(binaryPath +"vars.obj",'rb')
MAX_DESC_SEQUENCE_LENGTH, MAX_LOC_SEQUENCE_LENGTH, MAX_TEXT_SEQUENCE_LENGTH, MAX_NAME_SEQUENCE_LENGTH, MAX_TZ_SEQUENCE_LENGTH = pickle.load(file)

file = open(binaryPath +"data.obj",'rb')
trainDescription,  trainLocation, trainDomain, trainTld, trainSource, trainTexts, trainUserName, trainTZ, trainUtc, trainUserLang, trainCreatedAt, trainUserMentions = pickle.load(file)

trainSinTime = []
trainCosTime = []
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

def convertUsersToEmbeddings(trainUserMentions, embedding, idMapping):

    trainUser = np.zeros((len(trainUserMentions), embedding.shape[1]), dtype='float32')
    for i in range(len(trainUserMentions)):
        userMention = trainUserMentions[i]

        subkey = []
        for mention in userMention:
            if mention in idMapping.keys():
                subkey.append(idMapping[mention])

        if len(subkey) > 0:
            #bla = np.mean(embedding[subkey],axis=0)
            bla = embedding[subkey[0]]
            trainUser[i] = bla

    return trainUser

trainUser = convertUsersToEmbeddings(trainUserMentions, embedding, idMapping)

bla = trainUser.sum(axis=1)

#trainUser = trainUser[bla != 0,]
#classes = classes[bla != 0,]

userGraphBranchI = Input(shape=(trainUser.shape[1],), name="inputUserGraph")
userGraphBranch = BatchNormalization()(userGraphBranchI)
userGraphBranch = Dropout(0.2, name="domainName")(userGraphBranch)
userGraphBranchO = Dense(max(classes)+1, activation='softmax')(userGraphBranch)

userGraphModel= Model(inputs=userGraphBranchI, outputs=userGraphBranchO)
userGraphModel.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
sourceHistory = userGraphModel.fit(trainUser, classes,
                    epochs=nb_epoch, batch_size=batch_size,
                    verbose=verbosity
                    )
print("tldBranch finished after " +str(datetime.timedelta(seconds=round(time.time() - start))))
userGraphModel.save(modelPath + 'userGraphModel.h5')


