import gzip
import json
import random

"""
Generate a train test-split based on users 
"""

allUsers = set()
with gzip.open('/home/philippe/germanLanguage.json.gz','rb') as file: #7,615,371
    for line in file:
        parsed_json = json.loads(line.decode('utf-8'))
        allUsers.add(parsed_json['user']['id'])

print(len(allUsers)) #347,579
allUsers = list(allUsers)
seed = 1202
random.Random(seed).shuffle(allUsers)

trainUsers = allUsers[0:round(len(allUsers)/100*90)]
testUsers = allUsers[round(len(allUsers)/100*90):]

f1 = gzip.GzipFile("train.json.gz","wb") #6,032,664 lines
f2 = gzip.GzipFile("test.json.gz","wb") #643,299 lines

#f1 = open("train.json", "wb")
#f2 = open("test.json", "wb")

nLine = 0
with gzip.open('/home/philippe/germanLanguage.json.gz','rb') as file: #7,615,371
    for line in file:
        nLine+=1

        if nLine % 10000 == 0:
            print(nLine)

        parsed_json = json.loads(line.decode('utf-8'))

        if parsed_json['place']['place_type'] == 'city':

            userId = parsed_json['user']['id']
            if userId in trainUsers:
                f1.write(line)
            elif userId in testUsers:
                f2.write(line)
            else:
                print("Unknown split for '" +str(userId) +"'")

f1.close()
f2.close()