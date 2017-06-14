#Combine model and don't retrain individual trained models;

import pickle
from keras.models import load_model
from keras.layers import Dense, Merge
from keras.models import Sequential
import time

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

file = open("data/w-nut-latest/binaries/processors.obj",'rb')
descriptionTokenizer, linkTokenizer, locationTokenizer, sourceEncoder, textTokenizer, nameTokenizer, timeZoneTokenizer, utcEncoder, placeMedian, classes, colnames = pickle.load(file)

file = open("data/w-nut-latest/binaries/vars.obj",'rb')
MAX_DESC_SEQUENCE_LENGTH, MAX_URL_SEQUENCE_LENGTH, MAX_LOC_SEQUENCE_LENGTH, MAX_TEXT_SEQUENCE_LENGTH, MAX_NAME_SEQUENCE_LENGTH, MAX_TZ_SEQUENCE_LENGTH = pickle.load(file)

file = open("data/w-nut-latest/binaries/data.obj",'rb')
trainDescription, trainLinks, trainLocation, trainSource, trainTexts, trainUserName, trainTZ, trainUtc = pickle.load(file)

# create the model
batch_size = 256
nb_epoch = 3

#####################
# 9.) Merged model
from keras.models import Model
model1 = Model(input=descriptionBranch.input, output=descriptionBranch.get_layer('description').output)
model2 = Model(input=linkBranch.input, output=linkBranch.get_layer('link').output)
model3 = Model(input=locationBranch.input, output=locationBranch.get_layer('location').output)
model4 = Model(input=sourceBranch.input, output=sourceBranch.get_layer('source').output)
model5 = Model(input=textBranch.input, output=textBranch.get_layer('text').output)
model6 = Model(input=nameBranch.input, output=nameBranch.get_layer('username').output)
model7 = Model(input=tzBranch.input, output=tzBranch.get_layer('timezone').output)
model8 = Model(input=utcBranch.input, output=utcBranch.get_layer('utc').output)

for layer in model1.layers:
    layer.trainable = False

for layer in model2.layers:
    layer.trainable = False

for layer in model3.layers:
    layer.trainable = False

for layer in model5.layers:
    layer.trainable = False

for layer in model6.layers:
    layer.trainable = False

for layer in model7.layers:
    layer.trainable = False

merged = Merge([model1, model2, model3, model4, model5, model6, model7, model8], mode='concat', name="merged")
final_model = Sequential()
final_model.add(merged)
final_model.add(Dense(len(set(classes)), activation='softmax'))
final_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


start = time.time()
finalHistory = final_model.fit([trainDescription, trainLinks, trainLocation, trainSource, trainTexts, trainUserName, trainTZ, trainUtc],
                          classes,
                    nb_epoch=nb_epoch, batch_size=batch_size,
                    verbose=1
                    )
end = time.time()
print("final_model finished after " +str(end - start))

final_model.save('data/w-nut-latest/models/finalBranchTraible2.h5')