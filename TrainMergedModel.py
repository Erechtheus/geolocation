#Combine model and don't retrain trained models;
import pickle
import numpy as np
from keras.models import load_model
from keras.layers import LSTM, Bidirectional, Dropout, InputLayer, Dense, Merge, Reshape, Conv1D
from keras.models import Sequential
import time

#############################
# Load the eight individual models
descriptionBranch = load_model('data/w-nut-latest/models/descriptionBranchNorm.h5')
domainBranch = load_model('data/w-nut-latest/models/domainBranch.h5')
tldBranch = load_model('data/w-nut-latest/models/tldBranch.h5')
locationBranch = load_model('data/w-nut-latest/models/locationBranchNorm.h5')
sourceBranch = load_model('data/w-nut-latest/models/sourceBranch.h5')
textBranch = load_model('data/w-nut-latest/models/textBranchNorm.h5')
nameBranch = load_model('data/w-nut-latest/models/nameBranchNorm.h5')
tzBranch = load_model('data/w-nut-latest/models/tzBranchNorm.h5')
utcBranch = load_model('data/w-nut-latest/models/utcBranch.h5')
userLangBranch = load_model('data/w-nut-latest/models/userLangBranch.h5')
tweetTimeBranch = load_model('data/w-nut-latest/models/tweetTimeBranch.h5')

file = open("/media/philippe/5f695998-f5a5-4389-a2d8-4cf3ffa1288a/data/w-nut-latest/binaries/processors.obj",'rb')
descriptionTokenizer, domainEncoder, tldEncoder, locationTokenizer, sourceEncoder, textTokenizer, nameTokenizer, timeZoneTokenizer, utcEncoder, langEncoder, timeEncoder, placeMedian, classes, colnames = pickle.load(file)

file = open("/media/philippe/5f695998-f5a5-4389-a2d8-4cf3ffa1288a/data/w-nut-latest/binaries/vars.obj",'rb')
MAX_DESC_SEQUENCE_LENGTH, MAX_LOC_SEQUENCE_LENGTH, MAX_TEXT_SEQUENCE_LENGTH, MAX_NAME_SEQUENCE_LENGTH, MAX_TZ_SEQUENCE_LENGTH = pickle.load(file)

file = open("/media/philippe/5f695998-f5a5-4389-a2d8-4cf3ffa1288a/data/w-nut-latest/binaries/data.obj",'rb')
trainDescription,  trainLocation, trainDomain, trainTld, trainSource, trainTexts, trainUserName, trainTZ, trainUtc, trainUserLang, trainCreatedAt = pickle.load(file)

# create the model
batch_size = 256
nb_epoch = 3


##Convert data into one hot encodings
#2a
categorial = np.zeros((len(trainDomain), len(domainEncoder.classes_)), dtype="bool")
for i in range(len(trainDomain)):
    categorial[i, trainDomain[i]] = True
trainDomain = categorial

#2b
categorial = np.zeros((len(trainTld), len(tldEncoder.classes_)), dtype="bool")
for i in range(len(trainTld)):
    categorial[i, trainTld[i]] = True
trainTld = categorial

#4
categorial = np.zeros((len(trainSource), len(sourceEncoder.classes_)), dtype="bool")
for i in range(len(trainSource)):
    categorial[i, trainSource[i]] = True
trainSource = categorial

#8
categorial = np.zeros((len(trainUtc), len(utcEncoder.classes_)), dtype="bool")
for i in range(len(trainUtc)):
    categorial[i, trainUtc[i]] = True
trainUtc = categorial


#9) "User Language
categorial = np.zeros((len(trainUserLang), len(langEncoder.classes_)), dtype="bool")
for i in range(len(trainUserLang)):
    categorial[i, trainUserLang[i]] = True
trainUserLang = categorial

#10) #Tweet-Time (120)
categorial = np.zeros((len(trainCreatedAt), len(timeEncoder.classes_)), dtype="bool")
for i in range(len(trainCreatedAt)):
    categorial[i, trainCreatedAt[i]] = True
trainCreatedAt = categorial


#####################
# 11.) Merged model
from keras.models import Model
model1 = Model(inputs=descriptionBranch.input, outputs=descriptionBranch.get_layer('description').output)
model2a = Model(inputs=domainBranch.input, outputs=domainBranch.get_layer('domainName').output)
model2b = Model(inputs=tldBranch.input, outputs=tldBranch.get_layer('tld').output)
model3 = Model(inputs=locationBranch.input, outputs=locationBranch.get_layer('location').output)
model4 = Model(inputs=sourceBranch.input, outputs=sourceBranch.get_layer('source').output)
model5 = Model(inputs=textBranch.input, outputs=textBranch.get_layer('text').output)
model6 = Model(inputs=nameBranch.input, outputs=nameBranch.get_layer('username').output)
model7 = Model(inputs=tzBranch.input, outputs=tzBranch.get_layer('timezone').output)
model8 = Model(inputs=utcBranch.input, outputs=utcBranch.get_layer('utc').output)
model9 = Model(inputs=userLangBranch.input, outputs=userLangBranch.get_layer('userLang').output)
model10 = Model(inputs=tweetTimeBranch.input, outputs=tweetTimeBranch.get_layer('tweetTime').output)

for layer in model1.layers:
    layer.trainable = False

for layer in model2a.layers:
    layer.trainable = False

for layer in model2b.layers:
    layer.trainable = False

for layer in model3.layers:
    layer.trainable = False

for layer in model4.layers:
    layer.trainable = False

for layer in model5.layers:
    layer.trainable = False

for layer in model6.layers:
    layer.trainable = False

for layer in model7.layers:
    layer.trainable = False

for layer in model8.layers:
    layer.trainable = False

for layer in model9.layers:
    layer.trainable = False

for layer in model10.layers:
    layer.trainable = False


merged = Merge([model1, model2a, model2b, model3, model4, model5, model6, model7, model8, model9, model10], mode='concat', name="merged")
final_model = Sequential()
final_model.add(merged)
final_model.add(Dense(len(set(classes)), activation='softmax'))
final_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


start = time.time()
finalHistory = final_model.fit([trainDescription, trainDomain, trainTld, trainLocation, trainSource, trainTexts, trainUserName, trainTZ, trainUtc, trainUserLang, trainCreatedAt],
                          classes,
                    nb_epoch=nb_epoch, batch_size=batch_size,
                    verbose=1
                    )
end = time.time()
print("final_model finished after " +str(end - start))


model_yaml = final_model.to_yaml()
with open("data/w-nut-latest/models/finalmodel2.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
final_model.save_weights('data/w-nut-latest/models/finalmodelWeight2.h5')