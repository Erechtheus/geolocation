from keras.models import load_model
import pickle
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import tensorflow as tf
import os

from flask import Flask
from flask import request
from flask import json
from flask import Response

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

app = Flask(__name__)

binaryPath= 'data/binaries/'    #Place where the serialized training data is
modelPath= 'data/models/'       #Place to store the models

#Load text Model
textBranch = load_model(modelPath +'/textBranchNorm.h5')

graph = tf.get_default_graph()

#Load preprocessed data...
file = open(binaryPath +"processors.obj",'rb')
descriptionTokenizer, domainEncoder, tldEncoder, locationTokenizer, sourceEncoder, textTokenizer, nameTokenizer, timeZoneTokenizer, utcEncoder, langEncoder, placeMedian, colnames, classEncoder  = pickle.load(file)

#REST interface for European Language Grid
#Input: {'content': 'Montmartre is truly beautiful', 'mimeType': 'text/plain', 'type': 'text'}
#Response:{"response": {"classes": [{"class": "paris-a875-fr","score": 0.4133662283420563},{"class": "boulogne billancourt-a892-fr","score": 0.07033555209636688},{"class": "saint denis-a893-fr","score": 0.058008015155792236},{"class": "creteil-a894-fr","score": 0.02867915853857994},{"class": "argenteuil-a895-fr","score": 0.026361584663391113}],"type": "classification"}}
#Curl example:curl --header 'Content-Type: application/json' -X  POST --data-binary '{"type":"text", "content":"Montmartre is truly beautiful", "mimeType":"text/plain"}' 'http://127.0.0.1:8080/'
@app.route('/', methods = ['POST'])
def postJsonHandler():
    global graph
    with graph.as_default():

        content = request.get_json() #{'content': 'Montmartre is truly beautiful', 'mimeType': 'text/plain', 'type': 'text'}
        print("Content= '" +str(content) +"'")
        print("Content['content']= '" +str(content['content']) +"'")

        # Predict text (e.g., 'Montmartre is truly beautiful')
        testTexts = [];
        testTexts.append(content['content'])

        textSequences = textTokenizer.texts_to_sequences(testTexts)
        textSequences = np.asarray(textSequences)
        textSequences = pad_sequences(textSequences)

        predict = textBranch.predict(textSequences)

        #Sort predictions and take the highest ranked candidate
        # Print the 5
        maxCities = 5
        hits = []
        for index in reversed(predict.argsort()[0][-maxCities:]):
            print("%s with score=%.3f" % (colnames[index], float(predict[0][index])))
            my_dict = {
                'class': colnames[index],
                'score': float(predict[0][index])
            }
            hits.append(my_dict)
        x = {"response": {
                "type": "classification",
                "classes":hits
            }
        }
        print("hits")
        print(str(hits))
        return Response(json.dumps(x, indent=4), mimetype='application/json')

#Simple Rest Interface
#Predict text (e.g., 'Montmartre is truly beautiful')
#http://127.0.0.1:8080/predictText?text=Montmartre%20is%20truly%20beautiful
@app.route('/predictText')
def predictText():
    global graph
    with graph.as_default():
        text = request.args.get('text')
        print(text)
        maxCities = request.args.get('nResults', 5)

        # Predict text (e.g., 'Montmartre is truly beautiful')
        testTexts = [];
        testTexts.append(text)

        textSequences = textTokenizer.texts_to_sequences(testTexts)
        textSequences = np.asarray(textSequences)
        textSequences = pad_sequences(textSequences)

        predict = textBranch.predict(textSequences)

        # Print the topN
        hits = []
        for index in reversed(predict.argsort()[0][-maxCities:]):
            print("%s with score=%.3f" % (colnames[index], float(predict[0][index])))
            my_dict = {
                'city': colnames[index],
                'score': float(predict[0][index]),
                'lat': placeMedian[colnames[index]][0],
                'lon': placeMedian[colnames[index]][1]
            }
            hits.append(my_dict)
        x= {"query":text,
            "results":hits}
        print(hits)
        return Response(json.dumps(x, indent=4), mimetype='application/json')

if __name__ == '__main__':
   app.run(host="127.0.0.1", port=8080)