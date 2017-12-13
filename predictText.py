from keras.models import load_model
import pickle
from keras.preprocessing.sequence import pad_sequences
import numpy as np

#Load Model
textBranch = load_model('/media/philippe/5f695998-f5a5-4389-a2d8-4cf3ffa1288a/data/w-nut-latest/models/textBranchNorm.h5')

#Load tokenizers, and mapping
file = open("/media/philippe/5f695998-f5a5-4389-a2d8-4cf3ffa1288a/data/w-nut-latest/binaries/processors.obj",'rb')
descriptionTokenizer, linkTokenizer, locationTokenizer, sourceEncoder, textTokenizer, nameTokenizer, timeZoneTokenizer, utcEncoder, placeMedian, classes, colnames = pickle.load(file)

#Load properties from model
file = open("/media/philippe/5f695998-f5a5-4389-a2d8-4cf3ffa1288a/data/w-nut-latest/binaries/vars.obj",'rb')
MAX_DESC_SEQUENCE_LENGTH, MAX_URL_SEQUENCE_LENGTH, MAX_LOC_SEQUENCE_LENGTH, MAX_TEXT_SEQUENCE_LENGTH, MAX_NAME_SEQUENCE_LENGTH, MAX_TZ_SEQUENCE_LENGTH = pickle.load(file)

#Predict text (e.g., 'Montmartre is truly beautiful')
testTexts=[];
testTexts.append("Montmartre is truly beautiful")

textSequences = textTokenizer.texts_to_sequences(testTexts)
textSequences = np.asarray(textSequences)
textSequences = pad_sequences(textSequences, maxlen=MAX_TEXT_SEQUENCE_LENGTH)

predict = textBranch.predict(textSequences)

#Print the top 5
for index in reversed(predict.argsort()[0][-5:]):
    print("%s with score=%.3f" % (colnames[index], float(predict[0][index])) )
