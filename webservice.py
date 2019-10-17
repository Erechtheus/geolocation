from keras.engine.saving import model_from_yaml
from keras.models import load_model
import pickle
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import tensorflow as tf

from flask import Flask
from flask import request
from flask import json
from flask import Response

from representation import parseJsonLine, extractPreprocessUrl

app = Flask(__name__)


binaryPath= 'data/binaries/'    #Place where the serialized training data is
modelPath= 'data/models/'       #Place to store the models


#Load text Model
textBranch = load_model(modelPath +'/textBranchNorm.h5')

#load full json model
"""
yaml_file = open(modelPath +'finalmodel2.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
final_modelTrainable = model_from_yaml(loaded_model_yaml)
final_modelTrainable.load_weights(modelPath+"finalmodelWeight2.h5")
"""
graph = tf.get_default_graph()


#Load preprocessed data...
file = open(binaryPath +"processors.obj",'rb')
descriptionTokenizer, domainEncoder, tldEncoder, locationTokenizer, sourceEncoder, textTokenizer, nameTokenizer, timeZoneTokenizer, utcEncoder, langEncoder, placeMedian, colnames, classEncoder  = pickle.load(file)

#Predict text (e.g., 'Montmartre is truly beautiful')
#http://127.0.0.1:5000/predictText?text=Montmartre%20is%20truly%20beautiful
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

#Has some issues with json escape character //
"""
@app.route('/predictTweet')
def predictJson():
    global graph
    with graph.as_default():

        jsonLine = request.args.get('json')
        print(jsonLine)
        maxCities = request.args.get('nResults', 5)

        instance = parseJsonLine(jsonLine)

        testDescription = []; testLinks = []; testLocations = []; testSource = []; testTexts = []; testUserName = []; testTimeZone = []; testUtc = []; testUserLang = []; testSinTime = [];    testCosTime = []
        testDescription.append(str(instance.description))
        testLinks.append(extractPreprocessUrl(instance.urls))
        testLocations.append(str(instance.location))
        source = str(instance.source)
        testSource.append(source)
        testTexts.append(instance.text)
        testUserName.append(str(instance.name))
        testTimeZone.append(str(instance.timezone))
        testUtc.append(str(instance.utcOffset))
        testUserLang.append(str(instance.userLanguage))

        t = instance.createdAt.hour * 60 * 60 + instance.createdAt.minute * 60 + instance.createdAt.second
        t = 2 * np.pi * t / (24 * 60 * 60)
        testSinTime.append(np.sin(t))
        testCosTime.append(np.cos(t))

        ##Preprocessing
        # 1.) User-Description
        descriptionSequences = descriptionTokenizer.texts_to_sequences(testDescription)
        descriptionSequences = np.asarray(descriptionSequences)  # Convert to ndArray
        descriptionSequences = pad_sequences(descriptionSequences, maxlen=MAX_DESC_SEQUENCE_LENGTH)

        # 2.) Links
        # 2a.)
        testDomain = list(map(lambda x: x[0], testLinks))
        categorial = np.zeros((len(testDomain), len(domainEncoder.classes_)), dtype="bool")
        for i in range(len(testDomain)):
            if testDomain[i] in domainEncoder.classes_:
                categorial[i, domainEncoder.transform([testDomain[i]])[0]] = True
        testDomain = categorial

        # 2b)
        testTld = list(map(lambda x: x[1], testLinks))
        categorial = np.zeros((len(testTld), len(tldEncoder.classes_)), dtype="bool")
        for i in range(len(testTld)):
            if testTld[i] in tldEncoder.classes_:
                categorial[i, tldEncoder.transform([testTld[i]])[0]] = True
        testTld = categorial

        #3.) Location
        locationSequences = locationTokenizer.texts_to_sequences(testLocations)
        locationSequences = np.asarray(locationSequences)  # Convert to ndArray
        locationSequences = pad_sequences(locationSequences, maxlen=MAX_LOC_SEQUENCE_LENGTH)

        # 4.) Source
        categorial = np.zeros((len(testSource), len(sourceEncoder.classes_)), dtype="bool")
        for i in range(len(testSource)):
            if testSource[i] in sourceEncoder.classes_:
                categorial[i, sourceEncoder.transform([testSource[i]])[0]] = True
        testSource = categorial

        #5.) Text
        textSequences = textTokenizer.texts_to_sequences(testTexts)
        textSequences = np.asarray(textSequences)  # Convert to ndArray
        textSequences = pad_sequences(textSequences, maxlen=MAX_TEXT_SEQUENCE_LENGTH)

        # 6.) Username
        userSequences = nameTokenizer.texts_to_sequences(testUserName)
        userSequences = np.asarray(userSequences)  # Convert to ndArray
        userSequences = pad_sequences(userSequences, maxlen=MAX_NAME_SEQUENCE_LENGTH)

        #7.) TimeZone
        tzSequences = timeZoneTokenizer.texts_to_sequences(testTimeZone)
        tzSequences = np.asarray(tzSequences)  # Convert to ndArray
        tzSequences = pad_sequences(tzSequences, maxlen=MAX_TZ_SEQUENCE_LENGTH)

        # 8.) UTC
        categorial = np.zeros((len(testUtc), len(utcEncoder.classes_)), dtype="bool")
        for i in range(len(testUtc)):
            if testUtc[i] in utcEncoder.classes_:
                categorial[i, utcEncoder.transform([testUtc[i]])[0]] = True

        testUtc = categorial

        #9) "User Language
        categorial = np.zeros((len(testUserLang), len(langEncoder.classes_)), dtype="bool")
        for i in range(len(testUserLang)):
            if testUserLang[i] in langEncoder.classes_:
                categorial[i, langEncoder.transform([testUserLang[i]])[0]] = True

        predict = final_modelTrainable.predict(
            [descriptionSequences, testDomain, testTld, locationSequences, testSource, textSequences, userSequences,
             tzSequences, testUtc, testUserLang, np.column_stack((testSinTime, testCosTime))])

        # Print the topN
        result = []
        for index in reversed(predict.argsort()[0][-maxCities:]):
            print("%s with score=%.3f" % (colnames[index], float(predict[0][index])))
            my_dict = {
                'city': colnames[index],
                'score': float(predict[0][index]),
                'lat': placeMedian[colnames[index]][0],
                'lon': placeMedian[colnames[index]][1]
            }
            result.append(json.dumps(my_dict, indent=4))
        print(result)
        return Response(json.dumps(result, indent=4), mimetype='application/json')
"""

if __name__ == '__main__':
   app.run(host="127.0.0.1", port=5000)