from keras.models import load_model
import pickle
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from flask import Flask
from flask import request
from flask import json
from flask import Response
app = Flask(__name__)


binaryPath= 'data/binaries/'    #Place where the serialized training data is
modelPath= 'data/models/'       #Place to store the models


#Load Model
textBranch = load_model(modelPath +'/textBranchNorm.h5')

#Load tokenizers, and mapping
file = open(binaryPath +"processors.obj",'rb')
descriptionTokenizer, linkTokenizer, locationTokenizer, sourceEncoder, textTokenizer, nameTokenizer, timeZoneTokenizer, utcEncoder, placeMedian, classes, colnames = pickle.load(file)

#Load properties from model
file = open(binaryPath +"vars.obj",'rb')
MAX_DESC_SEQUENCE_LENGTH, MAX_URL_SEQUENCE_LENGTH, MAX_LOC_SEQUENCE_LENGTH, MAX_TEXT_SEQUENCE_LENGTH, MAX_NAME_SEQUENCE_LENGTH, MAX_TZ_SEQUENCE_LENGTH = pickle.load(file)


#Predict text (e.g., 'Montmartre is truly beautiful')
@app.route('/predictText', defaults={'nResults': 5})
def predictText():
   text = request.args.get('text')
   maxCities = request.args.get('nResults')
   testTexts=[];
   testTexts.append(text)

   textSequences = textTokenizer.texts_to_sequences(testTexts)
   textSequences = np.asarray(textSequences)
   textSequences = pad_sequences(textSequences, maxlen=MAX_TEXT_SEQUENCE_LENGTH)

   predict = textBranch.predict(textSequences)

   #Print the topN
   result = []
   for index in reversed(predict.argsort()[0][maxCities:]):
      print("%s with score=%.3f" % (colnames[index], float(predict[0][index])) )
      my_dict = {
       'city': colnames[index],
       'score': float(predict[0][index]),
       'lat': placeMedian[colnames[index]][0],  # 20.76
       'lon': placeMedian[colnames[index]][1]  # 69.07
      }
      result.append(json.dumps(my_dict))
   return Response(json.dump(result), mimetype='application/json')


#TODO Implement prediction of tweet
#@app.route('/predictTweet', defaults={'nResults': 5})


if __name__ == '__main__':
   app.run(host="127.0.0.1", port=5000)