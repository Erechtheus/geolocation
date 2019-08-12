#Load stuff:
#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

import gzip
import string
from collections import Counter

import numpy as np
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder

from representation import parseJsonLine, parseJsonLineWithPlace

binaryPath= 'data/binaries/'    #Place where the serialized training data is
modelPath= 'data/models/'       #Place to store the models
unknownClass = "unknownLocation" #place holder for unknown classes
trainFile="/home/philippe/Desktop/train.json.gz"
testFile="/home/philippe/Desktop/test.json.gz"



#2.) Tokenize texts
def my_filter():
    f = string.punctuation
    f += '\t\n\r…”'
    return f

textTokenizer = Tokenizer(num_words=200000, filters=my_filter(), oov_token='unknownToken')



def text_generator(trainingFile):
    with gzip.open(trainingFile,'rb') as file:
        for line in file:
            instance = parseJsonLine(line.decode('utf-8'))
            yield str(instance.text)

#Fit the textTokenizer to the training texts
textTokenizer.fit_on_texts(text_generator(trainFile))

"""
#Print some information from our tokenizer
new_dic = {}
for k,v in textTokenizer.word_counts.items():

    if v not in new_dic:
        new_dic[v] = []
    new_dic[v].append(k)

for key in sorted(new_dic.keys(), reverse=True)[0:100]:
    print(str(key) +"\t" +str(new_dic[key]))
"""

#Target class preparation
cnt = Counter()

classMap = {}
def class_generator(trainingFile):
    with gzip.open(trainingFile,'rb') as file:
        for line in file:
            instance = parseJsonLineWithPlace(line.decode('utf-8'))
            classMap[instance.place._name] =[instance.place.fullName, instance.place.country]
            yield str(instance.place._name)

#Fit target label to the counter object
cnt = Counter(class_generator(trainFile))

#Perform some basic analysis
cutoff = 25
print(str(len(cnt)) +" different classes")
filteredClasses = list(filter(lambda x: x[1] >=cutoff, list(cnt.items()))) #Remove all labels below a certain frequency (e.g., 25)
print("Classes with a frequency >= "+str(min(list(map(lambda x: x[1], filteredClasses))))  +"=" +str(len(filteredClasses)))
tmp=list(map(lambda x: x[0], filteredClasses))
tmp.append(unknownClass)
classEncoder = LabelEncoder()
classEncoder.fit(tmp)

print("Examples with known location " +str(np.sum(list(map(lambda x: x[1], filter(lambda x: x[1] >=cutoff, list(cnt.items())))))))
print("Examples with unknown location " +str(np.sum(list(map(lambda x: x[1], filter(lambda x: x[1] <cutoff, list(cnt.items())))))))

"""
averageSentence = []
with gzip.open(trainFile, 'rb') as file:
    for line in file:
        instance = parseJsonLineWithPlace(line.decode('utf-8'))
        averageSentence.append(len(textTokenizer.texts_to_sequences([instance.text])[0]))
np.median(averageSentence) #15 tokens
np.quantile(averageSentence, 0.75)#20.0 tokens
"""

MAX_TEXT_SEQUENCE_LENGTH = 20 #TODO; maybe perform no padding?

char2Idx = {}
for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|":
    char2Idx[c] = len(char2Idx) + 1 #+1 as 0 is masking


import pickle
filehandler = open(binaryPath + "germanVars.obj", "wb")
pickle.dump((classMap,classEncoder, textTokenizer, MAX_TEXT_SEQUENCE_LENGTH, unknownClass, char2Idx), filehandler)
filehandler.close()

