from keras.models import load_model
import pickle
from keras.preprocessing.sequence import pad_sequences
import numpy as np

binaryPath= 'data/binaries/'    #Place where the serialized training data is
modelPath= 'data/models/'       #Place to store the models

#Load Model
textBranch = load_model(modelPath +'/textBranchNorm.h5')

#Load preprocessed data...
file = open(binaryPath +"processors.obj",'rb')
descriptionTokenizer, domainEncoder, tldEncoder, locationTokenizer, sourceEncoder, textTokenizer, nameTokenizer, timeZoneTokenizer, utcEncoder, langEncoder, placeMedian, colnames, classEncoder  = pickle.load(file)

#Predict text (e.g., 'Montmartre is truly beautiful')
testTexts=[];
testTexts.append("Montmartre is truly beautiful")

textSequences = textTokenizer.texts_to_sequences(testTexts)
textSequences = np.asarray(textSequences)
textSequences = pad_sequences(textSequences)

predict = textBranch.predict(textSequences)

#Print the top 5
for index in reversed(predict.argsort()[0][-5:]):
    print("%s with score=%.3f" % (colnames[index], float(predict[0][index])) )
