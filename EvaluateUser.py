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
binaryPath= 'data/binaries/'    #Place where the serialized training data is
modelPath= 'data/models/'       #Place to store the models

testFile="data/test/data/test.user.json"
goldFile = 'data/test//test_labels/oracle.user.json'


# Load the eight individual models
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

#Retrained model
yaml_file = open(modelPath +'finalmodel.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
final_model = model_from_yaml(loaded_model_yaml)
final_model.load_weights(modelPath+"finalmodelWeight.h5")

#Retrained model
yaml_file = open(modelPath +'finalmodel2.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
final_modelTrainable = model_from_yaml(loaded_model_yaml)
final_modelTrainable.load_weights(modelPath+"finalmodelWeight2.h5")


##Evaluate model, this the the most likely place
def evalMax(predictions, type='USER', predictToFile='predictionsUserTmp.json'):
    out_file = open(predictToFile, "w")
    for userHash in set(testUserIds):
        indices = [i for i, x in enumerate(testUserIds) if x == userHash]

        prediction = predictions[indices] #Get all predictions for that user
        bestPlace = np.argmax(prediction) % prediction.shape[1]
        placeName = colnames[bestPlace]

        my_dict = {
            'hashed_user_id': userHash,
            'city': placeName,
            'lat': placeMedian[placeName][0],  # 20.76
            'lon': placeMedian[placeName][1]  # 69.07
        }
        # print(placeName +" " +instance.text)
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
testDescription = []; testLinks = []; testLocations=[]; testSource=[]; testTexts=[]; testUserName=[]; testTimeZone=[]; testUtc = [];  testUserIds=[]; testUserLang=[]; testSinTime = []; testCosTime=[]

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
    t = 2 * np.pi * t / (24 * 60 * 60)
    testSinTime.append(np.sin(t))
    testCosTime.append(np.cos(t))

    testUserIds.append(instance.userId)



#############################
#Convert the data

#############################
#Convert the data

#1.) User-Description
descriptionSequences = descriptionTokenizer.texts_to_sequences(testDescription)
descriptionSequences = np.asarray(descriptionSequences)  # Convert to ndArray
descriptionSequences = pad_sequences(descriptionSequences)

print("Description")
predict = descriptionBranch.predict(descriptionSequences)
evalMax(predict)        #/home/philippe/PycharmProjects/deepLearning/predictionsUser.json& USER& 0.097& 3407.9& 5896.8



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
print("Links")
predict = linkModel.predict(np.concatenate((testDomain, testTld), axis=1))
evalMax(predict)

#/home/philippe/PycharmProjects/deepLearning/predictionsUser.json& USER& 0.045& 6687.4& 6546.8


#3.) Location
locationSequences = locationTokenizer.texts_to_sequences(testLocations)
locationSequences = np.asarray(locationSequences)  # Convert to ndArray
locationSequences = pad_sequences(locationSequences)

print("Location")
predict = locationBranch.predict(locationSequences)
evalMax(predict)
#/home/philippe/PycharmProjects/deepLearning/predictionsUser.json& USER& 0.445& 43.9& 3831.7



#4.) Source
categorial = np.zeros((len(testSource), len(sourceEncoder.classes_)), dtype="bool")
for i in range(len(testSource)):
    if testSource[i] in sourceEncoder.classes_:
        categorial[i, sourceEncoder.transform([testSource[i]])[0]] = True
testSource = categorial

print("Source")
predict = sourceBranch.predict(categorial)
evalMax(predict)
#/home/philippe/PycharmProjects/deepLearning/predictionsUser.json& USER& 0.045& 6926.3& 6923.5



#5.) Text
textSequences = textTokenizer.texts_to_sequences(testTexts)
textSequences = np.asarray(textSequences)  # Convert to ndArray
textSequences = pad_sequences(textSequences)

print("Text")
predict = textBranch.predict(textSequences)
evalMax(predict)
#/home/philippe/PycharmProjects/deepLearning/predictionsUser.json& USER& 0.321& 263.8& 2570.9



#6.) Username
userSequences = nameTokenizer.texts_to_sequences(testUserName)
userSequences = np.asarray(userSequences)  # Convert to ndArray
userSequences = pad_sequences(userSequences)

print("Username")
predict = nameBranch.predict(userSequences)
evalMax(predict)
#/home/philippe/PycharmProjects/deepLearning/predictionsUser.json& USER& 0.059& 4140.4& 6107.6



#7.) TimeZone
tzSequences = timeZoneTokenizer.texts_to_sequences(testTimeZone)
tzSequences = np.asarray(tzSequences)  # Convert to ndArray
tzSequences = pad_sequences(tzSequences)

print("Timezone")
predict = tzBranch.predict(tzSequences)
evalMax(predict)        #/home/philippe/PycharmProjects/deepLearning/predictionsUser.json& USER& 0.062& 6926.3& 7270.9
#/home/philippe/PycharmProjects/deepLearning/predictionsUser.json& USER& 0.061& 5470.5& 5465.5


#8.) UTC
categorial = np.zeros((len(testUtc), len(utcEncoder.classes_)), dtype="bool")
for i in range(len(testUtc)):
    if testUtc[i] in utcEncoder.classes_:
        categorial[i, utcEncoder.transform([testUtc[i]])[0]] = True

print("UTC")
testUtc = categorial
predict = utcBranch.predict(testUtc)
evalMax(predict)

#/home/philippe/PycharmProjects/deepLearning/predictionsUser.json& USER& 0.051& 3883.4& 6422.6


#9
categorial = np.zeros((len(testUserLang), len(langEncoder.classes_)), dtype="bool")
for i in range(len(testUserLang)):
    if testUserLang[i] in langEncoder.classes_:
        categorial[i, langEncoder.transform([testUserLang[i]])[0]] = True

print("Userlang")
testUserLang = categorial
predict = userLangBranch.predict(testUserLang)
evalMax(predict)
#/home/philippe/PycharmProjects/deepLearning/predictionsUser.json& USER& 0.047& 8903.7& 8525.1

print("Tweet Time")
predict = tweetTimeBranch.predict(np.column_stack((testSinTime, testCosTime)))
evalMax(predict)
#/home/philippe/PycharmProjects/deepLearning/predictionsUser.json& USER& 0.024& 11720.6& 10363.2


#11.) Merged model
print("Full Model")
predict = final_model.predict([descriptionSequences, testDomain, testTld, locationSequences, testSource, textSequences, userSequences, tzSequences, testUtc, testUserLang, (np.column_stack((testSinTime, testCosTime)))])
evalMax(predict)
#/home/philippe/PycharmProjects/deepLearning/predictionsUser.json& USER& 0.53& 14.9& 838.5

#11.) Merged model
print("Full Model retrained")
predict = final_modelTrainable.predict([descriptionSequences, testDomain, testTld, locationSequences, testSource, textSequences, userSequences, tzSequences, testUtc, testUserLang, (np.column_stack((testSinTime, testCosTime)))])
evalMax(predict)
