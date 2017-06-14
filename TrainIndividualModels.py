#Load stuff:
import pickle
import time
from keras.models import Sequential
from keras.layers import LSTM, Dropout, InputLayer, Dense, Merge, BatchNormalization
from keras.layers.embeddings import Embedding


#Load preprocessed data...
file = open("data/w-nut-latest/binaries/processors.obj",'rb')
descriptionTokenizer, linkTokenizer, locationTokenizer, sourceEncoder, textTokenizer, nameTokenizer, timeZoneTokenizer, utcEncoder, placeMedian, classes, colnames = pickle.load(file)

file = open("w-nut-latest/binaries/vars.obj",'rb')
MAX_DESC_SEQUENCE_LENGTH, MAX_URL_SEQUENCE_LENGTH, MAX_LOC_SEQUENCE_LENGTH, MAX_TEXT_SEQUENCE_LENGTH, MAX_NAME_SEQUENCE_LENGTH, MAX_TZ_SEQUENCE_LENGTH = pickle.load(file)

file = open("data/w-nut-latest/binaries/data.obj",'rb')
trainDescription, trainLinks, trainLocation, trainSource, trainTexts, trainUserName, trainTZ, trainUtc = pickle.load(file)


##################Train
# Used settings
batch_size = 256
nb_epoch = 5
verbosity=2

descriptionEmbeddings = 100
linkEmbeddings = 100
locEmbeddings = 50
textEmbeddings = 100
nameEmbeddings = 100
tzEmbeddings = 50



####################
#1. Model: Description Model
descriptionBranch = Sequential()
descriptionBranch.add(Embedding(descriptionTokenizer.nb_words,
                                descriptionEmbeddings,
                                input_length=MAX_DESC_SEQUENCE_LENGTH,
                                dropout=0.2))
descriptionBranch.add(BatchNormalization())
descriptionBranch.add(Dropout(0.2))
descriptionBranch.add(LSTM(output_dim=30))
descriptionBranch.add(BatchNormalization())
descriptionBranch.add(Dropout(0.2, name="description"))

descriptionBranch.add(Dense(len(set(classes)), activation='softmax'))
descriptionBranch.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
descriptionHistory = descriptionBranch.fit(trainDescription, classes,
                    nb_epoch=nb_epoch, batch_size=batch_size,
                    verbose=verbosity
                    )
print("descriptionBranch finished after " +str(time.time() - start))
descriptionBranch.save('data/w-nut-latest/models/descriptionBranchNorm.h5')




#####################
#2. Model: Link Model
linkBranch = Sequential()
linkBranch.add(Embedding(linkTokenizer.nb_words,
                         linkEmbeddings,
                         input_length=MAX_URL_SEQUENCE_LENGTH,
                         mask_zero=True,
                         dropout=0.2))
linkBranch.add(BatchNormalization())
linkBranch.add(Dropout(0.2))
linkBranch.add(LSTM(output_dim=30))
linkBranch.add(BatchNormalization())
linkBranch.add(Dropout(0.2, name="link"))

linkBranch.add(Dense(len(set(classes)), activation='softmax'))
linkBranch.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
linkHistory = linkBranch.fit(trainLinks, classes,
                    nb_epoch=nb_epoch, batch_size=batch_size,
                    verbose=verbosity
                    )
print("linkBranch finished after " +str(time.time() - start))
linkBranch.save('data/w-nut-latest/models/linkBranchNorm.h5')





#####################
#3. Model: location Model
locationBranch = Sequential()
locationBranch.add(Embedding(locationTokenizer.nb_words,
                             locEmbeddings,
                    input_length=MAX_LOC_SEQUENCE_LENGTH,
                    dropout=0.2))
locationBranch.add(BatchNormalization())
locationBranch.add(Dropout(0.2))
locationBranch.add(LSTM(output_dim=30))
locationBranch.add(BatchNormalization())
locationBranch.add(Dropout(0.2, name="location"))

locationBranch.add(Dense(len(set(classes)), activation='softmax'))
locationBranch.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
locationHistory = locationBranch.fit(trainLocation, classes,
                    nb_epoch=nb_epoch, batch_size=batch_size,
                    verbose=verbosity
                    )
print("locationHistory finished after " +str(time.time() - start))
locationBranch.save('data/w-nut-latest/models/locationBranchNorm.h5')


#####################
#4. Model: Source Mode
sourceBranch = Sequential()
sourceBranch.add(InputLayer(input_shape=(trainSource.shape[1],), name="source"))

sourceBranch.add(Dense(len(set(classes)), activation='softmax'))
sourceBranch.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
sourceHistory = sourceBranch.fit(trainSource, classes,
                    nb_epoch=nb_epoch, batch_size=batch_size,
                    verbose=verbosity
                    )
print("sourceBranch finished after " +str(time.time() - start))
sourceBranch.save('data/w-nut-latest/models/sourceBranch.h5')



#####################
#5. Model: Text Model
textBranch = Sequential()
textBranch.add(Embedding(textTokenizer.nb_words,
                         textEmbeddings ,
                    input_length=MAX_TEXT_SEQUENCE_LENGTH,
                    dropout=0.2))
textBranch.add(BatchNormalization())
textBranch.add(Dropout(0.2))
textBranch.add(LSTM(output_dim=30))
textBranch.add(BatchNormalization())
textBranch.add(Dropout(0.2, name="text"))

textBranch.add(Dense(len(set(classes)), activation='softmax'))
textBranch.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
textHistory = textBranch.fit(trainTexts, classes,
                    nb_epoch=nb_epoch, batch_size=batch_size,
                    verbose=verbosity
                    )
print("textBranch finished after " +str(time.time() - start))
textBranch.save('data/w-nut-latest/models/textBranchNorm.h5')





#####################
# 6. Model: Name Model
nameBranch = Sequential()
nameBranch.add(Embedding(nameTokenizer.nb_words,
                         nameEmbeddings,
                         input_length=MAX_NAME_SEQUENCE_LENGTH,
                         dropout=0.2))
nameBranch.add(BatchNormalization())
nameBranch.add(Dropout(0.2))
nameBranch.add(LSTM(output_dim=30))
nameBranch.add(BatchNormalization())
nameBranch.add(Dropout(0.2, name="username"))

nameBranch.add(Dense(len(set(classes)), activation='softmax'))
nameBranch.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
nameHistory = nameBranch.fit(trainUserName, classes,
                    nb_epoch=nb_epoch, batch_size=batch_size,
                    verbose=verbosity
                    )
print("nameBranch finished after " +str(time.time() - start))
nameBranch.save('data/w-nut-latest/models/nameBranchNorm.h5')





#####################
# 7. Model: TimeZone Model
tzBranch = Sequential()
tzBranch.add(Embedding(timeZoneTokenizer.nb_words,
                       tzEmbeddings,
                       input_length=MAX_TZ_SEQUENCE_LENGTH,
                       dropout=0.2))
tzBranch.add(BatchNormalization())
tzBranch.add(Dropout(0.2))
tzBranch.add(LSTM(output_dim=30))
tzBranch.add(BatchNormalization())
tzBranch.add(Dropout(0.2, name="timezone"))

tzBranch.add(Dense(len(set(classes)), activation='softmax'))
tzBranch.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
tzHistory = tzBranch.fit(trainTZ, classes,
                    nb_epoch=nb_epoch, batch_size=batch_size,
                    verbose=verbosity
                    )
print("tzBranch finished after " +str(time.time() - start))
tzBranch.save('data/w-nut-latest/models/tzBranchNorm.h5')



#####################
# 8. Model: UTC Model
utcBranch = Sequential()
utcBranch.add(InputLayer(input_shape=(trainUtc.shape[1],), name="utc"))

utcBranch.add(Dense(len(set(classes)), activation='softmax'))
utcBranch.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
utcHistory = utcBranch.fit(trainUtc, classes,
                    nb_epoch=nb_epoch, batch_size=batch_size,
                    verbose=verbosity
                    )
print("utcBranch finished after " +str(time.time() - start))
utcBranch.save('data/w-nut-latest/models/utcBranch.h5')





#####################
# 9. Model: Merged model
from keras.models import Model
model1 = Model(input=descriptionBranch.input, output=descriptionBranch.get_layer('description').output)
model2 = Model(input=linkBranch.input, output=linkBranch.get_layer('link').output)
model3 = Model(input=locationBranch.input, output=locationBranch.get_layer('location').output)
model4 = Model(input=sourceBranch.input, output=sourceBranch.get_layer('source').output)
model5 = Model(input=textBranch.input, output=textBranch.get_layer('text').output)
model6 = Model(input=nameBranch.input, output=nameBranch.get_layer('username').output)
model7 = Model(input=tzBranch.input, output=tzBranch.get_layer('timezone').output)
model8 = Model(input=utcBranch.input, output=utcBranch.get_layer('utc').output)


merged = Merge([model1, model2, model3, model4, model5, model6, model7, model8], mode='concat', name="merged")
final_model = Sequential()
final_model.add(merged)
final_model.add(Dense(len(set(classes)), activation='softmax'))
final_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


start = time.time()
finalHistory = final_model.fit([trainDescription, trainLinks, trainLocation, trainSource, trainTexts, trainUserName, trainTZ, trainUtc],
                          classes,
                    nb_epoch=nb_epoch, batch_size=batch_size,
                    verbose=verbosity
                    )
end = time.time()
print("final_model finished after " +str(end - start))
final_model.save('ata/w-nut-latest/models/finalBranch.h5')
