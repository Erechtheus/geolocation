#Load stuff:
import os
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

#############################
# Load the eight individual models
binaryPath= 'data/binaries/'    #Place where the serialized training data is
modelPath= 'data/models/'       #Place to store the models

testFile="data/test/data/test.tweet.json"
goldFile='data/test/test_labels/oracle.tweet.json'

descriptionBranch = load_model(modelPath +'descriptionBranchNorm.h5')
linkModel = load_model(modelPath +'linkModel.h5') #Full link model
domainBranch = load_model(modelPath +'domainBranch.h5') #Partial link model
tldBranch = load_model(modelPath +'tldBranch.h5') #Partial link model
locationBranch = load_model(modelPath +'locationBranchNorm.h5')
sourceBranch = load_model(modelPath +'sourceBranch.h5')
textBranch = load_model(modelPath +'textBranchNorm.h5')
nameBranch = load_model(modelPath + 'nameBranchNorm.h5')
tzBranch = load_model(modelPath + 'tzBranchNorm.h5')
utcBranch = load_model(modelPath + 'utcBranch.h5')
userLangBranch = load_model(modelPath + 'userLangBranch.h5')
tweetTimeBranch = load_model(modelPath +'tweetTimeBranch.h5')

#Scratch Model:
yaml_file = open(modelPath +'finalmodel.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
final_model = model_from_yaml(loaded_model_yaml)
final_model.load_weights(modelPath +"finalmodelWeight.h5")


#Retrained model
yaml_file = open(modelPath +'finalmodel2.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
final_modelTrainable = model_from_yaml(loaded_model_yaml)
final_modelTrainable.load_weights(modelPath+"finalmodelWeight2.h5")


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


#############################
#Evaluate the models on the test data
file = open(binaryPath +"processors.obj",'rb')
descriptionTokenizer, domainEncoder, tldEncoder, locationTokenizer, sourceEncoder, textTokenizer, nameTokenizer, timeZoneTokenizer, utcEncoder, langEncoder, placeMedian, colnames, classEncoder  = pickle.load(file)

def roundMinutes(x, base=15):
    return int(base * round(float(x)/base))

##Load test-data
testDescription = []; testLinks = []; testLocations=[]; testSource=[]; testTexts=[]; testUserName=[]; testTimeZone=[]; testUtc = []; testIDs=[]; testUserLang=[];  testSinTime = []; testCosTime=[]
f = open(testFile)
for line in f:
    instance = parseJsonLine(line)

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
    t = 2*np.pi*t/(24*60*60)
    testSinTime.append(np.sin(t))
    testCosTime.append(np.cos(t))

    testIDs.append(instance.id)

#############################
#Convert the data

#1.) User-Description
descriptionSequences = descriptionTokenizer.texts_to_sequences(testDescription)
descriptionSequences = np.asarray(descriptionSequences)  # Convert to ndArray
descriptionSequences = pad_sequences(descriptionSequences)

predict = descriptionBranch.predict(descriptionSequences)
print("User description=")
eval(predict)
#/home/philippe/PycharmProjects/deepLearning/predictions.json& TWEET& 0.086& 3817.2& 6060.2


#2.) Links
#2a.)
testDomain = list(map(lambda x : x[0], testLinks))
categorial = np.zeros((len(testDomain), len(domainEncoder.classes_)), dtype="bool")
for i in range(len(testDomain)):
    if testDomain[i] in domainEncoder.classes_:
        categorial[i, domainEncoder.transform([testDomain[i]])[0]] = True
testDomain = categorial

#2b)
testTld = list(map(lambda x : x[1], testLinks))
categorial = np.zeros((len(testTld), len(tldEncoder.classes_)), dtype="bool")
for i in range(len(testTld)):
    if testTld[i] in tldEncoder.classes_:
        categorial[i, tldEncoder.transform([testTld[i]])[0]] = True
testTld = categorial

#2c)
print("User links")
predict = linkModel.predict(np.concatenate((testDomain, testTld), axis=1))
eval(predict)
#/home/philippe/PycharmProjects/deepLearning/predictions.json& TWEET& 0.032& 7601.7& 6980.5


#3.) Location
locationSequences = locationTokenizer.texts_to_sequences(testLocations)
locationSequences = np.asarray(locationSequences)  # Convert to ndArray
locationSequences = pad_sequences(locationSequences)

print("Location=")
predict = locationBranch.predict(locationSequences)
eval(predict)
#/home/philippe/PycharmProjects/deepLearning/predictions.json& TWEET& 0.361& 205.6& 4538.0




#4.) Source
categorial = np.zeros((len(testSource), len(sourceEncoder.classes_)), dtype="bool")
for i in range(len(testSource)):
    if testSource[i] in sourceEncoder.classes_:
        categorial[i, sourceEncoder.transform([testSource[i]])[0]] = True
testSource = categorial


print("Source=")
predict = sourceBranch.predict(testSource)
eval(predict)
#/home/philippe/PycharmProjects/deepLearning/predictions.json& TWEET& 0.045& 8005.0& 7516.8



#5.) Text
textSequences = textTokenizer.texts_to_sequences(testTexts)
textSequences = np.asarray(textSequences)  # Convert to ndArray
textSequences = pad_sequences(textSequences)

print("Text=")
predict = textBranch.predict(textSequences)
eval(predict)
#/home/philippe/PycharmProjects/deepLearning/predictions.json& TWEET& 0.195& 2190.6& 4472.9



#6.) Username
userSequences = nameTokenizer.texts_to_sequences(testUserName)
userSequences = np.asarray(userSequences)  # Convert to ndArray
userSequences = pad_sequences(userSequences)

print("Username=")
predict = nameBranch.predict(userSequences)
eval(predict)
#/home/philippe/PycharmProjects/deepLearning/predictions.json& TWEET& 0.057& 3849.0& 5930.1


#7.) TimeZone
tzSequences = timeZoneTokenizer.texts_to_sequences(testTimeZone)
tzSequences = np.asarray(tzSequences)  # Convert to ndArray
tzSequences = pad_sequences(tzSequences)

print("TimeZone=")
predict = tzBranch.predict(tzSequences)
eval(predict)
#/home/philippe/PycharmProjects/deepLearning/predictions.json& TWEET& 0.058& 5268.0& 5530.1



#8.) UTC
categorial = np.zeros((len(testUtc), len(utcEncoder.classes_)), dtype="bool")
for i in range(len(testUtc)):
    if testUtc[i] in utcEncoder.classes_:
        categorial[i, utcEncoder.transform([testUtc[i]])[0]] = True

testUtc = categorial

print("UTC=")
predict = utcBranch.predict(testUtc)
eval(predict)
#/home/philippe/PycharmProjects/deepLearning/predictions.json& TWEET& 0.046& 7698.1& 6849.0


#9) "User Language
categorial = np.zeros((len(testUserLang), len(langEncoder.classes_)), dtype="bool")
for i in range(len(testUserLang)):
    if testUserLang[i] in langEncoder.classes_:
        categorial[i, langEncoder.transform([testUserLang[i]])[0]] = True

testUserLang = categorial
print("Userlang=")
predict = userLangBranch.predict(testUserLang)
eval(predict)
#/home/philippe/PycharmProjects/deepLearning/predictions.json& TWEET& 0.061& 6465.1& 7310.2



#10) #Tweet-Time (120)
print("TweetTime")
"""
categorial = np.zeros((len(testCreatedAt), len(timeEncoder.classes_)), dtype="bool")
for i in range(len(testCreatedAt)):
    if testCreatedAt[i] in timeEncoder.classes_:
        categorial[i, timeEncoder.transform([testCreatedAt[i]])[0]] = True
    else:
        print("hmm  " +testCreatedAt[i])

testCreatedAt = categorial
"""
predict = tweetTimeBranch.predict(np.column_stack((testSinTime, testCosTime)))
eval(predict)
#/home/philippe/PycharmProjects/deepLearning/predictions.json& TWEET& 0.028& 8867.6& 8464.9




#11.) Merged model
print("Merged=")
predict = final_model.predict([descriptionSequences, testDomain, testTld, locationSequences, testSource, textSequences, userSequences, tzSequences, testUtc, testUserLang, np.column_stack((testSinTime, testCosTime))])
eval(predict)
#/home/philippe/PycharmProjects/deepLearning/predictions.json& TWEET& 0.417& 59.0& 1616.4



#12.) Merged model with original weights; without 2 parts which are not pretrained; maybe include?
print("Merged with full retraining=")
predict = final_modelTrainable.predict([descriptionSequences, testDomain, testTld, locationSequences, testSource, textSequences, userSequences, tzSequences, testUtc, testUserLang, np.column_stack((testSinTime, testCosTime)) ])
eval(predict)
#/home/philippe/PycharmProjects/deepLearning/predictions.json& TWEET& 0.43& 47.6& 1179.4
