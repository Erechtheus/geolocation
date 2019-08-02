import json
import unicodedata
from urllib.parse import urlparse
import re
import time
from datetime import datetime
import tldextract

class Place:
    def __init__(self, name=None, lat=None, lon=None, fullName=None, country = None):
        self._name = name
        self._lat = lat
        self._lon = lon
        self._fullName = fullName
        self._country = country

    @property
    def name(self):
        return self._name

    @property
    def lat(self):
        return self._lat

    @property
    def lon(self):
        return self._lon

    @property
    def fullName(self):
        return self._fullName

    @property
    def country(self):
        return self._country

    @name.setter
    def name(self, value):
        self._name = value

    @lat.setter
    def lat(self, value):
        self._lat = value

    @lon.setter
    def lon(self, value):
        self._lon = value

    @fullName.setter
    def fullName(self, value):
        self._fullName = value

    @country.setter
    def country(self, value):
        self._country = value

    def __str__(self):
        return 'ID=' +self._name +"\t" +"name= " +self._fullName +"-" +self._country

class Media:
    def __init__(self, url=None, type=None):
        self._url = url
        self._type = type

        @property
        def url(self):
            return self._url

        # photo, animated_gif, video
        @property
        def type(self):
            return self._type

        @url.setter
        def url(self, value):
            self._url = value

        @type.setter
        def type(self, value):
            self._type = value

#Python representation of a tweet
class Instance:
    def __init__(self, text="", place="", timezone=None, utcOffset=None, location=None, source=None, media=None, id=None, urls=None, name=None, description = None, userId=None, userLanguage=None, createdAt =None, userMentions=None):
        self._text = text               # The text of the tweet
        self._place = place             # The place if known
        self._timezone = timezone       # Timezone as string (e.g., Hawaii, Baghdad, Pacific Time, ...)
        self._utcOffset = utcOffset     # UTC-Offset (e.g., 10800, ...)
        self._location = location       # User location (e.g., Paradise, City of dreams, '', France)
        self._source = source           # Twitter for Android...
        self._media = media             # Media information
        self._id = id                   # ID
        self._urls = urls               # URLS
        self._name = name               # Username using user self description
        self._description = description #Description
        self._userId = userId           # Username using user unique twitter ID
        self._userLanguage= userLanguage #he user’s self-declared user interface language.
        self._createdAt = createdAt     #When has the tweet been sent?
        self._userMentions = userMentions #IDs of other userMentions

    def __str__(self):
        return "Tweet '" + str(self._id) +"' userId ='" + str(self._userId) +"' usermentions='" +str(self._userMentions) #+"' text = '" + str(self._text) + "'"

    @property
    def text(self):
        return self._text

    @property
    def place(self):
        return self._place

    @property
    def timezone(self):
        return self._timezone

    @property
    def utcOffset(self):
        return self._utcOffset

    @property
    def location(self):
        return self._location

    @property
    def source(self):
        return self._source

    @property
    def media(self):
        return self._media

    @property
    def id(self):
        return self._id

    @property
    def urls(self):
        return self._urls

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    @property
    def userId(self):
        return self._userId

    @property
    def createdAt(self):
        return self._createdAt

    @property
    def userLanguage(self):
        return self._userLanguage

    @property
    def userMentions(self):
        return self._userMentions

    @text.setter
    def text(self, value):
        self._text = value

    @place.setter
    def place(self, value):
        self._place = value

    @timezone.setter
    def timezone(self, value):
        self._timezone = value

    @utcOffset.setter
    def utcOffset(self, value):
        self._utcOffset = value

    @location.setter
    def location(self, value):
        self._location = value

    @source.setter
    def source(self, value):
        self._source = value

    @media.setter
    def media(self, value):
        self._media = value

    @id.setter
    def id(self, value):
        self._id = value

    @urls.setter
    def urls(self, value):
        self._urls = value

    @name.setter
    def name(self, value):
        self._name = value

    @description.setter
    def description(self, value):
        self._description = value

    @userId.setter
    def userId(self, value):
        self._userId = value

    @userLanguage.setter
    def userLanguage(self, value):
        self._userLanguage = value

    @createdAt.setter
    def createdAt(self, value):
        self._createdAt = value

    @userMentions.setter
    def userMentions(self, value):
        self._userMentions = value



#Parses Twitter JSON and returns an Istance object
def parseJsonLine( line ):

    tweet = json.loads(line)
    #normalized = unicodedata.normalize('NFKD', tweet['text']).encode('ASCII', 'ignore').decode('UTF-8') 
    normalized = tweet['text']
    tweetTime = datetime.strptime(tweet['created_at'], '%a %b %d %H:%M:%S +0000 %Y') #
    instance = Instance(text=normalized, timezone=tweet['user']['time_zone'],  location=tweet['user']['location'], name=tweet['user']['name'],
                        utcOffset = tweet['user']['utc_offset'], description = tweet['user']['description'], userLanguage = tweet['user']['lang'], createdAt = tweetTime)

    #Convert UTC from hours to seconds
    utcOffset = tweet['user']['utc_offset']
    if(utcOffset != None):
        utcOffset = utcOffset/60/60 #We convert it into hours representation
    instance._utcOffset = utcOffset

    #Save either twitter ID or for test set hashed-id
    if('id' in tweet):
        instance._id =  tweet['id']
    elif('hashed_tweet_id' in tweet):
        instance._id = tweet['hashed_tweet_id']
    else:
        print("No twitter id")

    if ('id_str' in tweet):
        instance._userId = tweet['user']['id_str']
    elif('hashed_user_id' in tweet):
        instance._userId = tweet['hashed_user_id']
    else:
        print("No user id")

    mentions = []
    userMentions = tweet['entities']['user_mentions']
    for mention in userMentions:
        mentions.append(mention['id_str'])
    instance.userMentions = mentions

    #Extract relevant information from source
    source = tweet['source']
    if (source != ''):
        source = source[2 + source.index("\">"):source.index("</a>")]
        instance.source = source.strip()

    #Extract Media-Information in tweets (e.g., fotos, videos, ...)
    if ('extended_entities' in tweet):
        instance.media = Media(url=tweet['extended_entities']['media'][0]['media_url'], type=tweet['extended_entities']['media'][0]['type'])

    #Extract URL's in tweet
    #Entities are: urls, user_mentions, symbols, hashtags
    if('entities' in tweet and len(tweet['entities']['urls']) > 0):
        instance.urls = tweet['entities']['urls']

    return instance



#Parses Twitter JSON and returns an Istance object
def parseJsonLineWithPlace( line ):

    tweet = json.loads(line)


    placeName = (tweet['place']['id'])
    placeFullName =tweet['place']['full_name']
    placeCountry = tweet['place']['country_code']
    placeLon = None
    placeLat = None
    if(tweet['coordinates'] != None):
        placeLon = float(tweet['coordinates']['coordinates'][0])
        placeLat = float(tweet['coordinates']['coordinates'][1])
    place = Place(name=placeName, lat=placeLat, lon=placeLon, fullName=placeFullName, country=placeCountry)

    instance = Instance(text=tweet['text'], place=place, timezone=tweet['user']['time_zone'],  location=tweet['user']['location'], name=tweet['user']['name'], utcOffset = tweet['user']['utc_offset'], description = tweet['user']['description'])

    #Convert UTC from hours to seconds
    utcOffset = tweet['user']['utc_offset']
    if(utcOffset != None):
        utcOffset = utcOffset/60/60 #We convert it into hours representation
    instance._utcOffset = utcOffset

    #Save either twitter ID or for test set hashed-id
    if('id' in tweet):
        instance._id =  tweet['id']
    elif('hashed_tweet_id' in tweet):
        instance._id = tweet['hashed_tweet_id']

    #Extract relevant information from source
    source = tweet['source']
    if (source != ''):
        source = source[2 + source.index("\">"):source.index("</a>")]
        instance.source = source.strip()

    #Extract Media-Information in tweets (e.g., fotos, videos, ...)
    if ('extended_entities' in tweet):
        instance.media = Media(url=tweet['extended_entities']['media'][0]['media_url'], type=tweet['extended_entities']['media'][0]['type'])

    #Extract URL's in tweet
    #Entities are: urls, user_mentions, symbols, hashtags
    if('entities' in tweet and len(tweet['entities']['urls']) > 0):
        instance.urls = tweet['entities']['urls']

    return instance



"""
Provide a relatively accurate center lat, lon returned as a list pair, given
a list of list pairs.
ex: in: geolocations = ((lat1,lon1), (lat2,lon2),)
    out: (center_lat, center_lon)
"""
def center_geolocation(geolocations):
    from math import cos, sin, atan2, sqrt, radians, degrees

    x = 0
    y = 0
    z = 0

    for lat, lon in geolocations:
        lat = radians(float(lat))
        lon = radians(float(lon))
        x += cos(lat) * cos(lon)
        y += cos(lat) * sin(lon)
        z += sin(lat)

    x = float(x / len(geolocations))
    y = float(y / len(geolocations))
    z = float(z / len(geolocations))

    return ( degrees(atan2(z, sqrt(x * x + y * y))), degrees(atan2(y, x)))


"""
Given a url (e.g., http://www.google.de/abcdf?fx=search), return only domain  and remove leading www (e.g., google.de)
"""
def extractPreprocessUrl(url):
    if(url == None):
        return (str(''),str(''))

    elif(type(url) is str):
        tmp = tldextract.extract(url)
        return (tmp.domain.lower(), tmp.suffix.lower()) # Return a tupple (e.g., instagram/com, facebook/de)

    else:
        tmp = tldextract.extract(url[0]['expanded_url'])
        return (tmp.domain.lower(), tmp.suffix.lower())

