#Load stuff:
#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

import datetime
import math
import time

from keras import Input
from keras import Model
from keras.layers import Dropout, Dense, BatchNormalization, LSTM
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate

from customModels.generatorUtils import batch_generator

binaryPath= 'data/binaries/'    #Place where the serialized training data is
modelPath= 'data/models/'       #Place to store the models
trainFile="/home/philippe/Desktop/train.json.gz"
testFile="/home/philippe/Desktop/test.json.gz"


##################Train
# create the model
batch_size = 256
nb_epoch = 5
verbosity=2


textEmbeddings = 100


import pickle

file = open(binaryPath +"germanVars.obj",'rb')
classMap,classEncoder, textTokenizer, MAX_TEXT_SEQUENCE_LENGTH, unknownClass, char2Idx = pickle.load(file)



inputs = []; concat = []

#</Text Model>
textBranchI = Input(shape=(None,), name="inputText")
inputs.append(textBranchI)
textBranch = Embedding(textTokenizer.num_words,
                         textEmbeddings,
                        #input_length=MAX_TEXT_SEQUENCE_LENGTH,
                        mask_zero=True
                         )(textBranchI)
textBranch = BatchNormalization()(textBranch)
textBranch = Dropout(0.2)(textBranch)

concat.append(textBranch)
#</Text Model>

##<Char Embedding>
"""
char_ids = Input(batch_shape=(None, None, None), dtype='int32', name='char_input')
inputs.append(char_ids)
char_embeddings = Embedding(input_dim=len(char2Idx)+1,
                            output_dim=25,
                            mask_zero=false,
                            name='char_embedding')(char_ids)
char_embeddings = TimeDistributed(Conv1D(self._char_filter_size, self._char_filter_length, padding='same'),
                                  name="char_cnn")(char_embeddings)
char_embeddings = TimeDistributed(GlobalMaxPooling1D(), name="char_pooling")(char_embeddings)
concat.append(char_embeddings)
"""
##</Char Embedding>


#Build concatenated layer
if len(concat) >= 2:
    concatenated = concatenate(concat)
else:
    concatenated = concat[0]

textBranch = LSTM(units=100, dropout=0.2, recurrent_dropout=0.2)(concatenated)
textBranch = BatchNormalization()(textBranch)
#textBranch = Dropout(0.2, name="text")(textBranch)
textBranchO = Dense(len(classEncoder.classes_), activation='softmax')(textBranch)





textModel = Model(inputs=textBranchI, outputs=textBranchO)
textModel.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

import subprocess
stdoutdata = subprocess.getoutput("zcat " +trainFile +" | wc --lines ")
numberOfTrainingSamples=int(stdoutdata)

stdoutdata = subprocess.getoutput("zcat " +testFile +" | wc --lines ")
numberOfTestingSamples=int(stdoutdata)



start = time.time()
textHistory = textModel.fit_generator(generator=batch_generator(twitterFile=trainFile, classEncoder=classEncoder, textTokenizer=textTokenizer, maxlen=MAX_TEXT_SEQUENCE_LENGTH, char2Idx=char2Idx, unknownClass=unknownClass,  batch_size=batch_size), steps_per_epoch=math.ceil(numberOfTrainingSamples / batch_size), #numberOfTrainingSamples
                                      validation_data=batch_generator(twitterFile=testFile, classEncoder=classEncoder, textTokenizer=textTokenizer, maxlen=MAX_TEXT_SEQUENCE_LENGTH, char2Idx=char2Idx, unknownClass=unknownClass, batch_size=batch_size), validation_steps=math.ceil(numberOfTestingSamples/batch_size), #numberOfTestingSamples
                                      epochs=nb_epoch, verbose=verbosity
                                      )
print("textBranch finished after " +str(datetime.timedelta(seconds=round(time.time() - start))))
textModel.save(modelPath +'germanText.h5')



