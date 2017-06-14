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

#Load final models consisting of submodels
final_model = load_model('data/w-nut-latest/models/finalBranch.h5') #Trained from scratch
final_modelTrainable = load_model('data/w-nut-latest/models/finalBranchTraible2.h5') #Model trained by reusing the individual models



#Function to evaluate individual models
def eval(predictions, type='TWEET', predictToFile='predictions.json', goldFile='test/test_labels/oracle.tweet.json'):
    out_file = open(predictToFile, "w")
    for i in range(predictions.shape[0]):
        id = testIDs[i]
        prediction = predictions[i]
        bestPlace = np.argmax(prediction)
        placeName = colnames[bestPlace]

        my_dict = {
            'hashed_tweet_id': id,
            'city': placeName,
            'lat': placeMedian[placeName][0],
            'lon': placeMedian[placeName][1]
        }
        # print(placeName +" " +instance.text)
        json.dump(my_dict, out_file)
        out_file.write("\n")
    out_file.close()
    evaluate_submission(predictToFile, goldFile, type)


#############################
#Load tokenizers and other information
file = open("data/w-nut-latest/binaries/processors.obj",'rb')
descriptionTokenizer, linkTokenizer, locationTokenizer, sourceEncoder, textTokenizer, nameTokenizer, timeZoneTokenizer, utcEncoder, placeMedian, classes, colnames = pickle.load(file)

file = open("data/w-nut-latest/binaries/vars.obj",'rb')
MAX_DESC_SEQUENCE_LENGTH, MAX_URL_SEQUENCE_LENGTH, MAX_LOC_SEQUENCE_LENGTH, MAX_TEXT_SEQUENCE_LENGTH, MAX_NAME_SEQUENCE_LENGTH, MAX_TZ_SEQUENCE_LENGTH = pickle.load(file)

##Load test-data
testDescription = []; testLinks = []; testLocations=[]; testSource=[]; testTexts=[]; testUserName=[]; testTimeZone=[]; testUtc = []; testIDs=[]
testFile="data/w-nut-latest/test/data/test.tweet.json"
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

    testIDs.append(instance.id)


#############################
#Predict and eval the individual models

#1.) User-Description
descriptionSequences = descriptionTokenizer.texts_to_sequences(testDescription)
descriptionSequences = np.asarray(descriptionSequences)
descriptionSequences = pad_sequences(descriptionSequences, maxlen=MAX_DESC_SEQUENCE_LENGTH)

predict = descriptionBranch.predict(descriptionSequences)
eval(predict)
#/home/philippe/PycharmProjects/deepLearning/predictions.json& TWEET& 0.087& 3806.7& 6048.9


#2.) Links
testLinks = linkTokenizer.texts_to_sequences(testLinks)
testLinks = np.asarray(testLinks)
testLinks = pad_sequences(testLinks, maxlen=MAX_URL_SEQUENCE_LENGTH)

predict = linkBranch.predict(testLinks)
eval(predict)
#/home/philippe/PycharmProjects/deepLearning/predictions.json& TWEET& 0.033& 7593.4& 6978.6


#3.) Location
locationSequences = locationTokenizer.texts_to_sequences(testLocations)
locationSequences = np.asarray(locationSequences)
locationSequences = pad_sequences(locationSequences, maxlen=MAX_LOC_SEQUENCE_LENGTH)

predict = locationBranch.predict(locationSequences)
eval(predict)
#/home/philippe/PycharmProjects/deepLearning/predictions.json& TWEET& 0.362& 209.4& 4535.7


#4.) Source
testSource = sourceEncoder.transform(testSource)
categorial = np.zeros((len(testSource), len(sourceEncoder.classes_)-1))
for i in range(len(testSource)):
    categorial[i, testSource[i]] = 1

predict = sourceBranch.predict(categorial)
eval(predict)
#/home/philippe/PycharmProjects/deepLearning/predictions.json& TWEET& 0.044& 8029.0& 7528.2


#5.) Text
textSequences = textTokenizer.texts_to_sequences(testTexts)
textSequences = np.asarray(textSequences)
textSequences = pad_sequences(textSequences, maxlen=MAX_TEXT_SEQUENCE_LENGTH)

predict = textBranch.predict(textSequences)
eval(predict)
#/home/philippe/PycharmProjects/deepLearning/predictions.json& TWEET& 0.193& 2128.4& 4404.3





#6.) Username
userSequences = nameTokenizer.texts_to_sequences(testUserName)
userSequences = np.asarray(userSequences)
userSequences = pad_sequences(userSequences, maxlen=MAX_NAME_SEQUENCE_LENGTH)

predict = nameBranch.predict(userSequences)
eval(predict)
#/home/philippe/PycharmProjects/deepLearning/predictions.json& TWEET& 0.059& 3942.5& 5990.1



#7.) TimeZone
tzSequences = timeZoneTokenizer.texts_to_sequences(testTimeZone)
tzSequences = np.asarray(tzSequences)
tzSequences = pad_sequences(tzSequences, maxlen=MAX_TZ_SEQUENCE_LENGTH)

predict = tzBranch.predict(tzSequences)
eval(predict)
#/home/philippe/PycharmProjects/deepLearning/predictions.json& TWEET& 0.062& 6504.1& 7144.1


#8.) UTC
testUtc = utcEncoder.transform(testUtc)
testUtc = np_utils.to_categorical(testUtc)

predict = utcBranch.predict(testUtc)
eval(predict)
#/home/philippe/PycharmProjects/deepLearning/predictions.json& TWEET& 0.05& 6610.3& 7191.9



#9.) Merged model
predict = final_model.predict([descriptionSequences, testLinks, locationSequences, categorial, textSequences, userSequences, tzSequences, testUtc])
eval(predict)
#/home/philippe/PycharmProjects/deepLearning/predictions.json& TWEET& 0.417& 59.0& 1616.4


#10.) Merged model with original weights; without 2 parts which are not pretrained; maybe include?
predict = final_modelTrainable.predict([descriptionSequences, testLinks, locationSequences, categorial, textSequences, userSequences, tzSequences, testUtc])
eval(predict)
#/home/philippe/PycharmProjects/deepLearning/predictions.json& TWEET& 0.414& 54.5& 1477.8

