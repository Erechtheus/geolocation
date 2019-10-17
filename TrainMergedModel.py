#Combine model and don't retrain trained models;
import pickle

import datetime
import numpy as np
from keras.models import load_model
from keras.layers import Dense, concatenate, BatchNormalization
import time

import os
from sklearn.utils import shuffle
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

#############################
binaryPath= 'data/binaries/'    #Place where the serialized training data is
modelPath= 'data/models/'       #Place to store the models

#Random seed
#from numpy.random import seed
#seed(2019*2*5)
#from tensorflow import set_random_seed
#set_random_seed(2019*2*5)

# Load the eight individual models
descriptionBranch = load_model(modelPath +'descriptionBranchNorm.h5')
domainBranch = load_model(modelPath +'domainBranch.h5')
tldBranch = load_model(modelPath +'tldBranch.h5')
locationBranch = load_model(modelPath +'locationBranchNorm.h5')
sourceBranch = load_model(modelPath +'sourceBranch.h5')
textBranch = load_model(modelPath +'textBranchNorm.h5')
nameBranch = load_model(modelPath +'nameBranchNorm.h5')
tzBranch = load_model(modelPath +'tzBranchNorm.h5')
utcBranch = load_model(modelPath +'utcBranch.h5')
userLangBranch = load_model(modelPath +'userLangBranch.h5')
tweetTimeBranch = load_model(modelPath +'tweetTimeBranch.h5')

#Load preprocessed data...
file = open(binaryPath +"processors.obj",'rb')
descriptionTokenizer, domainEncoder, tldEncoder, locationTokenizer, sourceEncoder, textTokenizer, nameTokenizer, timeZoneTokenizer, utcEncoder, langEncoder, placeMedian, colnames, classEncoder  = pickle.load(file)

file = open(binaryPath +"data.obj",'rb')
trainDescription, trainLocation, trainDomain, trainTld, trainSource, trainTexts, trainUserName, trainTZ, trainUtc, trainUserLang, trainCreatedAt, classes= pickle.load(file)

#Shuffle train-data
trainDescription,  trainLocation, trainDomain, trainTld, trainSource, trainTexts, trainUserName, trainTZ, trainUtc, trainUserLang, trainCreatedAt, classes = shuffle(trainDescription,  trainLocation, trainDomain, trainTld, trainSource, trainTexts, trainUserName, trainTZ, trainUtc, trainUserLang, trainCreatedAt, classes, random_state=1202)


# create the model
batch_size = 256
nb_epoch = 3
validation_split = 0.01 #91279 samples for validation


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


mergedBranch = concatenate([model1.output, model2a.output, model2b.output, model3.output, model4.output, model5.output, model6.output, model7.output, model8.output, model9.output, model10.output], axis=-1)
mergedBranch = BatchNormalization(name="finalNorm")(mergedBranch)
predictions = Dense(len(set(classes)), activation='softmax', name="predictions")(mergedBranch)

final_model = Model(inputs=[model1.input, model2a.input, model2b.input, model3.input, model4.input, model5.input, model6.input, model7.input, model8.input, model9.input, model10.input], outputs=predictions)
final_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


start = time.time()
finalHistory = final_model.fit([trainDescription, trainDomain, trainTld, trainLocation, trainSource, trainTexts, trainUserName, trainTZ, trainUtc, trainUserLang, trainCreatedAt],
                          classes,
                    epochs=nb_epoch, batch_size=batch_size,
                    verbose=2, validation_split=validation_split
                    )
end = time.time()
print("final_model finished after " +str(datetime.timedelta(seconds=time.time() - start)))


model_yaml = final_model.to_yaml()
with open(modelPath +"finalmodel.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
final_model.save_weights(modelPath +'finalmodelWeight.h5')

#########################
for layer in final_model.layers:
    layer.trainable = True
    final_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

start = time.time()
finalHistory = final_model.fit([trainDescription, trainDomain, trainTld, trainLocation, trainSource, trainTexts, trainUserName, trainTZ, trainUtc, trainUserLang, trainCreatedAt],
                          classes,
                    epochs=nb_epoch, batch_size=batch_size,
                    verbose=2, validation_split=validation_split
                    )
end = time.time()
print("final_model finished after " +str(datetime.timedelta(seconds=time.time() - start)))


model_yaml = final_model.to_yaml()
with open(modelPath +"finalmodel2.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
final_model.save_weights(modelPath +'finalmodelWeight2.h5')
