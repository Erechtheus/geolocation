#Load stuff:
import pickle
import numpy as np
import json
from representation import parseJsonLine, extractPreprocessUrl
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from geoEval import evaluate_submission

#############################
# Load the eight individual models
descriptionBranch = load_model('data/w-nut-latest/models/descriptionBranchNorm.h5')
linkBranch = load_model('data/w-nut-latest/models/linkBranchNorm.h5')
locationBranch = load_model('data/w-nut-latest/models/locationBranchNorm.h5')
sourceBranch = load_model('data/w-nut-latest/models/sourceBranch.h5')
textBranch = load_model('data/w-nut-latest/models/textBranchNorm.h5')
nameBranch = load_model('data/w-nut-latest/models/nameBranchNorm.h5')
tzBranch = load_model('data/w-nut-latest/models/tzBranchNorm.h5')
utcBranch = load_model('data/w-nut-latest/models/utcBranch.h5')

#Load final models
final_model = load_model('data/w-nut-latest/models/finalBranch.h5')
final_modelTrainable = load_model('data/w-nut-latest/models/finalBranchTraible2.h5') #Saved weights


##Evaluate model, this the the most likely place
def evalMax(predictions, type='USER', predictToFile='/home/philippe/PycharmProjects/deepLearning/predictionsUser.json', goldFile='data/w-nut-latest/test/test_labels/oracle.user.json'):
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
file = open("data/w-nut-latest/binaries/processors.obj",'rb')
descriptionTokenizer, linkTokenizer, locationTokenizer, sourceEncoder, textTokenizer, nameTokenizer, timeZoneTokenizer, utcEncoder, placeMedian, classes, colnames = pickle.load(file)

#TODO: Probably we can perform prediction without these files; based on the model definition
file = open("data/w-nut-latest/binaries/vars.obj",'rb')
MAX_DESC_SEQUENCE_LENGTH, MAX_URL_SEQUENCE_LENGTH, MAX_LOC_SEQUENCE_LENGTH, MAX_TEXT_SEQUENCE_LENGTH, MAX_NAME_SEQUENCE_LENGTH, MAX_TZ_SEQUENCE_LENGTH = pickle.load(file)

##Load test-data
testDescription = []; testLinks = []; testLocations=[]; testSource=[]; testTexts=[]; testUserName=[]; testTimeZone=[]; testUtc = [];  testUserIds=[]
testFile="data/w-nut-latest/test/data/test.user.json"
f = open(testFile)
for line in f:
    instance = parseJsonLine(line)

    testDescription.append(str(instance.description))
    testLinks.append(extractPreprocessUrl(instance.urls))
    testLocations.append(str(instance.location))

    source = str(instance.source)
    if(source not in set(sourceEncoder.classes_)):
        source = "unknown"
    testSource.append(source)
    testTexts.append(instance.text)
    testUserName.append(str(instance.name))
    testTimeZone.append(str(instance.timezone))
    if (instance.utcOffset == -4.5): #Change UTC offsets
        instance.utcOffset = -5.0
    testUtc.append(str(instance.utcOffset))

    testUserIds.append(instance.userName)


#############################
#Convert the data

#1.) User-Description
descriptionSequences = descriptionTokenizer.texts_to_sequences(testDescription)
descriptionSequences = np.asarray(descriptionSequences)  # Convert to ndArray
descriptionSequences = pad_sequences(descriptionSequences, maxlen=MAX_DESC_SEQUENCE_LENGTH)

predict = descriptionBranch.predict(descriptionSequences)
evalMax(predict)        #/home/philippe/PycharmProjects/deepLearning/predictionsUser.json& USER& 0.097& 3407.9& 5896.8

#2.) Links
testLinks = linkTokenizer.texts_to_sequences(testLinks)
testLinks = np.asarray(testLinks) #Convert to ndArray
testLinks = pad_sequences(testLinks, maxlen=MAX_URL_SEQUENCE_LENGTH)

predict = linkBranch.predict(testLinks)
evalMax(predict)        #/home/philippe/PycharmProjects/deepLearning/predictionsUser.json& USER& 0.045& 6732.0& 6554.3


#3.) Location
locationSequences = locationTokenizer.texts_to_sequences(testLocations)
locationSequences = np.asarray(locationSequences)  # Convert to ndArray
locationSequences = pad_sequences(locationSequences, maxlen=MAX_LOC_SEQUENCE_LENGTH)

predict = locationBranch.predict(locationSequences)
evalMax(predict)        #/home/philippe/PycharmProjects/deepLearning/predictionsUser.json& USER& 0.441& 45.9& 3841.8


#4.) Source
testSource = sourceEncoder.transform(testSource)
categorial = np.zeros((len(testSource), len(sourceEncoder.classes_)-1))
for i in range(len(testSource)):
    categorial[i, testSource[i]] = 1

predict = sourceBranch.predict(categorial)
evalMax(predict)        #/home/philippe/PycharmProjects/deepLearning/predictionsUser.json& USER& 0.045& 6950.8& 6938.5


#5.) Text
textSequences = textTokenizer.texts_to_sequences(testTexts)
textSequences = np.asarray(textSequences)  # Convert to ndArray
textSequences = pad_sequences(textSequences, maxlen=MAX_TEXT_SEQUENCE_LENGTH)

predict = textBranch.predict(textSequences)
evalMax(predict)        #/home/philippe/PycharmProjects/deepLearning/predictionsUser.json& USER& 0.322& 266.4& 2595.0


#6.) Username
userSequences = nameTokenizer.texts_to_sequences(testUserName)
userSequences = np.asarray(userSequences)  # Convert to ndArray
userSequences = pad_sequences(userSequences, maxlen=MAX_NAME_SEQUENCE_LENGTH)

predict = nameBranch.predict(userSequences)
evalMax(predict)        #/home/philippe/PycharmProjects/deepLearning/predictionsUser.json& USER& 0.058& 4153.4& 6116.0


#7.) TimeZone
tzSequences = timeZoneTokenizer.texts_to_sequences(testTimeZone)
tzSequences = np.asarray(tzSequences)  # Convert to ndArray
tzSequences = pad_sequences(tzSequences, maxlen=MAX_TZ_SEQUENCE_LENGTH)

predict = tzBranch.predict(tzSequences)
evalMax(predict)        #/home/philippe/PycharmProjects/deepLearning/predictionsUser.json& USER& 0.062& 6926.3& 7270.9


#8.) UTC
testUtc = utcEncoder.transform(testUtc)
testUtc = np_utils.to_categorical(testUtc)

predict = utcBranch.predict(testUtc)
evalMax(predict)        #/home/philippe/PycharmProjects/deepLearning/predictionsUser.json& USER& 0.05& 6530.9& 7211.7


#9.) Merged model
predict = final_model.predict([descriptionSequences, testLinks, locationSequences, categorial, textSequences, userSequences, tzSequences, testUtc])
evalMax(predict)        #/home/philippe/PycharmProjects/deepLearning/predictionsUser.json& USER& 0.513& 17.8& 1023.9



#10.) Merged model with original weights; without 2 parts which are not pretrained; maybe include?
predict = final_modelTrainable.predict([descriptionSequences, testLinks, locationSequences, categorial, textSequences, userSequences, tzSequences, testUtc])
evalMax(predict)        #/home/philippe/PycharmProjects/deepLearning/predictionsUser.json& USER& 0.524& 15.9& 916.1
