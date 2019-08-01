#Load stuff:
#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

import datetime
import gzip
import string
import time
from collections import Counter

import math
import numpy as np
from keras import Input
from keras import Model
from keras.layers import Dropout, Dense, BatchNormalization, SpatialDropout1D, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder

from representation import parseJsonLine, parseJsonLineWithPlace

binaryPath= 'data/binaries/'    #Place where the serialized training data is
modelPath= 'data/models/'       #Place to store the models
trainFile="/home/philippe/PycharmProjects/geolocation/train.json.gz"
testFile="/home/philippe/PycharmProjects/geolocation/test.json.gz"


##################Train
# create the model
batch_size = 256
nb_epoch = 5
verbosity=2


textEmbeddings = 100
unknownClass = "unknownLocation" #place holder for unknown classes


import pickle

file = open(modelPath +"germanVars.obj",'rb')
classMap,classEncoder, textTokenizer, MAX_TEXT_SEQUENCE_LENGTH = pickle.load(file)

char2Idx = {}
for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|":
    char2Idx[c] = len(char2Idx) + 1 #+1 as 0 is masking

def docs2chars(docs, char2Idx):

    #We create a three dimensional tensor with
    #Number of samples; Max number of tokens; Max number of characters
    nSamples = len(docs)                                            #Number of samples
    maxTokensSentences = max([len(x) for x in docs])                #Max token per sentence
    maxCharsToken = max([max([len(y) for y in x]) for x in docs])     #Max chars per token

    x = np.zeros((nSamples,
                  maxTokensSentences,
                  maxCharsToken
                  )).astype('int32')#probably int32 is to large

    for i, doc in enumerate(docs):
        for j, token in enumerate(doc):
            tokenRepresentation = [char2Idx.get(char, len(char2Idx) + 1) for char in token]
            x[i, j, :len(tokenRepresentation)] = tokenRepresentation

    return(x)







def batch_generator(twitterFile, batch_size=64):
    with gzip.open(twitterFile, 'rb') as file:
        trainTexts = []; trainLabels=[]
        for line in file:

            #Reset after each batch
            if len(trainTexts) == batch_size:
                trainTexts = []; trainLabels=[]

            instance = parseJsonLineWithPlace(line.decode('utf-8'))
            trainTexts.append(instance.text)

            if instance.place._name not in classEncoder.classes_:
                trainLabels.append(unknownClass)
            else:
                trainLabels.append(instance.place._name)

            if len(trainTexts) == batch_size:

                #Character embeddings
                trainCharacters = docs2chars(trainTexts, char2Idx=char2Idx)

                #Text Tweet
                trainTexts = textTokenizer.texts_to_sequences(trainTexts)
                trainTexts = np.asarray(trainTexts)  # Convert to ndArraytop
                trainTexts = pad_sequences(trainTexts, maxlen=MAX_TEXT_SEQUENCE_LENGTH)


                # class label
                classes = classEncoder.transform(trainLabels)

                #yield trainTexts, classes
                yield ({
                        'inputText' : trainTexts,
                        },
                       classes
                       )
        #Return the last batch of instances after reaching end of file...
        if len(trainTexts) > 0:
            trainTexts = textTokenizer.texts_to_sequences(trainTexts)
            trainTexts = np.asarray(trainTexts)  # Convert to ndArraytop
            trainTexts = pad_sequences(trainTexts, maxlen=MAX_TEXT_SEQUENCE_LENGTH)

            # class label
            classes = classEncoder.transform(trainLabels)
            yield ({
                       'inputText': trainTexts,
                   },
                   classes
            )



#Text Model
#inputs = []; concat = []
textBranchI = Input(shape=(None,), name="inputText")
#inputs.append(textBranchI)
textBranch = Embedding(textTokenizer.num_words,
                         textEmbeddings,
                        #input_length=MAX_TEXT_SEQUENCE_LENGTH,
                        mask_zero=True
                         )(textBranchI)
#textBranch = SpatialDropout1D(rate=0.2)(textBranch)
textBranch = BatchNormalization()(textBranch)
#textBranch = Dropout(0.2)(textBranch)
textBranch = LSTM(units=100, dropout=0.2, recurrent_dropout=0.2)(textBranch)
textBranch = BatchNormalization()(textBranch)
#textBranch = Dropout(0.2, name="text")(textBranch)
textBranchO = Dense(len(classEncoder.classes_), activation='softmax')(textBranch)


##<Char Embedding>
"""
char_ids = Input(batch_shape=(None, None, None), dtype='int32', name='char_input')
inputs.append(char_ids)
char_embeddings = Embedding(input_dim=self._char_vocab_size,
                            output_dim=self._char_embedding_dim,
                            mask_zero=self._use_char_lstm,
                            name='char_embedding')(char_ids)
char_embeddings = TimeDistributed(Conv1D(self._char_filter_size, self._char_filter_length, padding='same'),
                                  name="char_cnn")(char_embeddings)
char_embeddings = TimeDistributed(GlobalMaxPooling1D(), name="char_pooling")(char_embeddings)

concat.append(char_embeddings)
"""
##</Char Embedding>



textModel = Model(inputs=textBranchI, outputs=textBranchO)
textModel.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

import subprocess
stdoutdata = subprocess.getoutput("zcat " +trainFile +" | wc --lines ")
numberOfTrainingSamples=int(stdoutdata)

stdoutdata = subprocess.getoutput("zcat " +testFile +" | wc --lines ")
numberOfTestingSamples=int(stdoutdata)



start = time.time()
textHistory = textModel.fit_generator(generator=batch_generator(twitterFile=trainFile, batch_size=batch_size), steps_per_epoch=math.ceil(numberOfTrainingSamples / batch_size), #numberOfTrainingSamples
                                      validation_data=batch_generator(twitterFile=testFile, batch_size=batch_size), validation_steps=math.ceil(numberOfTestingSamples/batch_size), #numberOfTestingSamples
                                      epochs=nb_epoch, verbose=verbosity
                                      )
print("textBranch finished after " +str(datetime.timedelta(seconds=round(time.time() - start))))
textModel.save(modelPath +'germanText.h5')



"""
###################
#Let's test this shit :)
from keras.models import load_model
import pickle

modelPath= 'data/models/'       #Place to store the models
file = open(modelPath +"germanVars.obj",'rb')
classMap,classEncoder, textTokenizer = pickle.load(file)

textModel = load_model(modelPath +'germanText.h5')
for prediction in textModel.predict_generator(generator=batch_generator(twitterFile=testFile, batch_size=batch_size),steps = math.ceil(numberOfTestingSamples/batch_size), verbose=1):
    print(prediction)
"""

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

predictions = []
groundTruths = []
i = 0
for batch in batch_generator(twitterFile=testFile, batch_size=batch_size):
    print(i)
    i = i +1
    tmp = textModel.predict_on_batch(batch[0])

    prediction = np.argmax(tmp, axis=1)
    predictions.append(prediction)
    groundTruth = batch[1]
    groundTruths.append(groundTruth)

matrix = confusion_matrix(groundTruths, predictions)

accuracy_score(groundTruth, prediction)
#precision_recall_fscore_support(groundTruth, prediction)
#multilabel_confusion_matrix(groundTruth, prediction)
import matplotlib.pyplot as plt
plt.imshow(matrix, cmap='binary', interpolation='None')
plt.show()