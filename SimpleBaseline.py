#Weak baseline, which predicts the most frequently observed training class and the mean

import gzip
import json

import numpy as np

from representation import parseJsonLine, Place

trainingFile="data/train/training.twitter.json.gz" #File with  all ~9 Million training tweets
placesFile='data/train/training.json.gz'           #Place annotation provided by task organisers

testTweet="data/test/test.tweet.json"              # WNUT test file for tweets
testUser= "data/test/test.user.json"               # WNUT test file for User

#Parse Twitter-JSON
tweetToTextMapping= {} # Map<Twitter-ID; tweet>
with gzip.open(trainingFile,'rb') as file:
    for line in file:
        instance = parseJsonLine(line.decode('utf-8'))
        tweetToTextMapping[instance.id] = instance

#Parse and add gold-label for tweets
with gzip.open(placesFile,'rb') as file:
    for line in file:
        parsed_json = json.loads(line.decode('utf-8'))
        tweetId=int(parsed_json["tweet_id"])
        if(tweetId in tweetToTextMapping):
            place = Place(name=parsed_json["tweet_city"], lat=parsed_json["tweet_latitude"], lon=parsed_json["tweet_longitude"])
            tweetToTextMapping[tweetId].place = place

#Extract all place gold-labels for all tweets
places = list(map(lambda  x : x.place, tweetToTextMapping.values()))

#Check which element occurs most frequently
# Most frequent element is 'jakarta-04-id' with 279,192 tweets
from collections import Counter
counts = Counter(map(lambda  x : x._name, places))

#Find the average location for Jakarta
jakarta = list(filter(lambda  x : x._name ==  'jakarta-04-id', places))
lat = np.mean(list(map(lambda x: float(x._lat), jakarta)))
lon = np.mean(list(map(lambda x: float(x._lon), jakarta)))


###Evaluate simple baseline for Tweet predictions
f = open(testTweet)

testIDs=[]
for line in f:
    instance = parseJsonLine(line)
    testIDs.append(instance.id)


out_file = open("predictions.json","w")
for id in testIDs:
    my_dict = {
        'hashed_tweet_id': id,
        'city': 'jakarta-04-id',
        'lat': lat,  # 20.76
        'lon': lon  # 69.07
    }
    # print(placeName +" " +instance.text)
    json.dump(my_dict, out_file)
    out_file.write("\n")

out_file.close()

from geoEval import evaluate_submission
evaluate_submission('predictions.json', 'test/test_labels/oracle.tweet.json', 'TWEET')


###Evaluate simple baseline for User predictions
f = open(testUser)

testUsernames=set()
for line in f:
    instance = parseJsonLine(line)
    testUsernames.add(instance.userName)

out_file = open('predictionsUser.json', "w")
for userHash in testUsernames:
    my_dict = {
        'hashed_user_id': userHash,
        'city':  'jakarta-04-id',
        'lat': lat,
        'lon': lon
    }
    json.dump(my_dict, out_file)
    out_file.write("\n")
out_file.close()

from geoEval import evaluate_submission
evaluate_submission('predictionsUser.json', 'test/test_labels/oracle.user.json', 'USER')