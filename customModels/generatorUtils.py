import gzip

import numpy as np
from keras.preprocessing.sequence import pad_sequences

from representation import parseJsonLineWithPlace


def docs2chars(docs, char2Idx):

    #We create a three dimensional tensor with
    #Number of samples; Max number of tokens; Max number of characters
    nSamples = len(docs)                                            #Number of samples
    maxTokensSentences = max([len(x) for x in docs])                #Max token per sentence
    maxCharsToken = max([max([len(y) for y in x]) for x in docs])     #Max chars per token

    x = np.zeros((nSamples,
                  maxTokensSentences,
                  maxCharsToken
                  )).astype('int32')#probably int32 is to large

    for i, doc in enumerate(docs):
        for j, token in enumerate(doc):
            tokenRepresentation = [char2Idx.get(char, len(char2Idx) + 1) for char in token]
            x[i, j, :len(tokenRepresentation)] = tokenRepresentation

    return(x)




def batch_generator(twitterFile, classEncoder, textTokenizer, char2Idx, maxlen, unknownClass="unk", batch_size=64):
    while True:
        print("Opening file '" +twitterFile  +"'")
        with gzip.open(twitterFile, 'rb') as file:
            trainTexts = []; trainLabels=[]
            for line in file:

                #Reset after each batch
                if len(trainTexts) == batch_size:
                    trainTexts = []; trainLabels=[]

                instance = parseJsonLineWithPlace(line.decode('utf-8'))
                trainTexts.append(instance.text)

                if instance.place._name not in classEncoder.classes_:
                    trainLabels.append(unknownClass)
                else:
                    trainLabels.append(instance.place._name)

                if len(trainTexts) == batch_size:

                    #Character embeddings
                    trainCharacters = docs2chars(trainTexts, char2Idx=char2Idx)

                    #Text Tweet
                    trainTexts = textTokenizer.texts_to_sequences(trainTexts)
                    trainTexts = np.asarray(trainTexts)  # Convert to ndArraytop
                    trainTexts = pad_sequences(trainTexts, maxlen=maxlen)


                    # class label
                    classes = classEncoder.transform(trainLabels)

                    #yield trainTexts, classes
                    yield ({
                            'inputText' : trainTexts,
                            'char_embedding' : trainCharacters,
                            },
                           classes
                           )
            print("Reached end of generator; usualle End-of-epoch")
            #Return the last batch of instances after reaching end of file...
            if len(trainTexts) > 0:
                trainTexts = textTokenizer.texts_to_sequences(trainTexts)
                trainTexts = np.asarray(trainTexts)  # Convert to ndArraytop
                trainTexts = pad_sequences(trainTexts, maxlen=maxlen)

                # class label
                classes = classEncoder.transform(trainLabels)
                yield ({
                           'inputText': trainTexts,
                       },
                       classes
                )
