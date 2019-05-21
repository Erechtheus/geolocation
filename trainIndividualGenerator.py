#Load stuff:
#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pickle
import numpy as np
import time
from keras.layers import Dropout, Dense, BatchNormalization, SpatialDropout1D, LSTM, concatenate
from keras.layers.embeddings import Embedding
import math
import datetime
from keras import Input
from keras import Model

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import numpy as np
import gzip
import json
from representation import parseJsonLine, Place, extractPreprocessUrl
from tensorflow.python.keras.utils import Sequence
from collections import Counter
import pickle
import gc
import string


#Random seed
from numpy.random import seed
seed(2019*2*5)
from tensorflow import set_random_seed
set_random_seed(2019*2*5)

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
verbosity=1

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
            trainDescriptions = []
            trainLabels = []
            for line in file:

                if len(trainDescriptions) == batch_size:
                    trainDescriptions = []
                    trainLabels = []

                instance = parseJsonLine(line.decode('utf-8'))

                trainDescription = str(instance.description)
                trainDescriptions.append(trainDescription)

                trainLabel = goldstandard[instance.id]._name
                trainLabels.append(trainLabel)

                #print(str(instance.id) +"\t" +str(len(trainDescriptions)))

                if len(trainDescriptions) == batch_size:

                    #Descriptions
                    trainDescriptions = descriptionTokenizer.texts_to_sequences(trainDescriptions)
                    trainDescriptions = np.asarray(trainDescriptions)  # Convert to ndArraytop
                    trainDescriptions = pad_sequences(trainDescriptions, maxlen=MAX_DESC_SEQUENCE_LENGTH)

                    # class label
                    classes = classEncoder.transform(trainLabels)

                    yield trainDescriptions, classes
                    # yield ({'inputDescription': trainDescriptions, 'input_2': x2}, {'output': y}) #TODO Use this in order to produce all relevant parts


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
                    epochs=nb_epoch, samples_per_epoch=math.ceil(numberOfTrainingsamples/batch_size),
                    verbose=verbosity
                    )
print("descriptionBranch finished after " +str(datetime.timedelta(seconds=round(time.time() - start))))
descriptionModel.save(modelPath +'descriptionBranchNorm.h5')



"""
class mygenerator(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def 

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        # read your data here using the batch lists, batch_x and batch_y
#        x = [my_readfunction(filename) for filename in batch_x]
#        y = [my_readfunction(filename) for filename in batch_y]
#        return np.array(x), np.array(y)

trainGenerator = mygenerator(x_set=trainingFile, y_set=idToGold)
trainGenerator.__getitem__(10)
"""