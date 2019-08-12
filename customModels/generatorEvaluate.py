import subprocess

import math
import pickle

from keras.models import load_model
from os import uname

from customModels.generatorUtils import batch_generator

binaryPath= '../data/binaries/'    #Place where the serialized training data is
modelPath= '../data/models/'       #Place to store the models
testFile="/home/philippe/Desktop/test.json.gz"

file = open(binaryPath +"germanVars.obj",'rb')
classMap,classEncoder, textTokenizer, MAX_TEXT_SEQUENCE_LENGTH, unknownClass, char2Idx = pickle.load(file)


stdoutdata = subprocess.getoutput("zcat " +testFile +" | wc --lines ")
numberOfTestingSamples=int(stdoutdata)

batch_size=64


textModel = load_model(modelPath +'germanText.h5')
for prediction in textModel.predict_generator(generator=batch_generator(twitterFile=testFile, batch_size=batch_size),steps = math.ceil(numberOfTestingSamples/batch_size), verbose=1):
    print(prediction)

batch_generator(twitterFile=testFile, classEncoder=classEncoder, textTokenizer=textTokenizer, maxlen=MAX_TEXT_SEQUENCE_LENGTH,
                char2Idx=char2Idx, unknownClass=uname(), batch_size=batch_size, validation_steps = math.ceil( numberOfTestingSamples / batch_size))

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

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