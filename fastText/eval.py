#Load stuff:
import os
import string

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pickle
import numpy as np
import json
from representation import parseJsonLine, extractPreprocessUrl
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from geoEval import evaluate_submission
from keras.models import model_from_yaml
from gensim.models.fasttext import FastText


#############################
# Load the eight individual models
binaryPath= 'data/binaries/'    #Place where the serialized training data is
modelPath= 'data/models/'       #Place to store the models

testFile="data/test/data/test.tweet.json"
goldFile='data/test/test_labels/oracle.tweet.json'




textBranch = load_model(modelPath +'GeneratortextBranchNorm.h5')

file = open(binaryPath + "GeneratorProcessors.obj",'rb')
placeMedian, classEncoder   = pickle.load(file)


fastTextModel = "/media/philippe/5f695998-f5a5-4389-a2d8-4cf3ffa1288a/embeddings/crawl-300d-2M-subword/crawl-300d-2M-subword.bin"
model = FastText.load_fasttext_format(fastTextModel)

#2.) Tokenize texts
def my_filter():
    f = string.punctuation
    f += '\t\n\r…”'
    return f


translate_map = dict((ord(c), "") for c in my_filter())


###Extract relevant parts and store in list...
def getDocumentVectors(text, model):
    tokens = text.split()
    docRep = []
    for token in tokens:
        token = token.lower()
        token = token.translate(translate_map)
        vector = model[token]
        docRep.append(vector)
    return np.asarray(docRep)

colnames = list(classEncoder.classes_)

##Load test-data
testIDs=[]; testTexts=[]
f = open(testFile)
for line in f:
    instance = parseJsonLine(line)

    testTexts.append(getDocumentVectors(instance.text, model))
    testIDs.append(instance.id)

##Evaluate model
def eval(predictions, type='TWEET', predictToFile='predictionsTmp.json'):
    out_file = open(predictToFile, "w")
    for i in range(predictions.shape[0]):
        id = testIDs[i]
        prediction = predictions[i]
        bestPlace = np.argmax(prediction)
        placeName = colnames[bestPlace]

        my_dict = {
            'hashed_tweet_id': id,
            'city': placeName,
            'lat': placeMedian[placeName][0],  # 20.76
            'lon': placeMedian[placeName][1]  # 69.07
        }
        json.dump(my_dict, out_file)
        out_file.write("\n")
    out_file.close()
    evaluate_submission(predictToFile, goldFile, type)



#1.) User-Description
testTexts = np.asarray(testTexts)
testTexts = pad_sequences(testTexts, dtype = 'float32')

predict = textBranch.predict(testTexts)
print("User description=")
eval(predict)

#units = 30
# predictionsTmp.json& TWEET& 0.132& 2796.8& 4807.9
#unit = 100
#predictionsTmp.json& TWEET& 0.166& 2457.5& 4491.3
#units=100, recurrent_dropout=0.2
#predictionsTmp.json& TWEET& 0.164& 2458.1& 4468.2

#units=300, recurrent_dropout=0.2
#predictionsTmp.json& TWEET& 0.183& 2184.0& 4256.7
