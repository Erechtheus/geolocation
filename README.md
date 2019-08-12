# geolocation
Geolocation prediction for a given Tweet, or a short text. The system trains a neural net, as described in

	Philippe Thomas and Leonhard Hennig (2017), "Twitter Geolocation Prediction using Neural Networks." In Proceedings of GSCL

# Usage

## Download 
Source code from this repository has been published here (https://github.com/Erechtheus/geolocation/releases). 
- [Version 1.0](https://github.com/Erechtheus/geolocation/releases/tag/V1.0) refers to the original code, which has been presented in our GSCL paper. 
- [Version 2.1](https://github.com/Erechtheus/geolocation/releases/tag/V2.1)  (recommended) uses keras functional API (instead of Keras sequential API). This code runs with Keras Version 2, whereas the original release worked with Keras Version 1. Also has some minor improvement regarding preprocessing and has a REST-API.

## Train and apply models
To train models, training data (tweets and gold labels) needs to be retrieved. As Tweets can not be shared directly, we refer to the [WNUT'16 workshop page](http://noisy-text.github.io/2016/geo-shared-task.html) for further information.

After retrieving the training files, the [preprocess](https://github.com/Erechtheus/geolocation/blob/master/Preprocess.py) script converts tweets into the desired representation to train a neural network. Models can be trained from scratch using the [trainindividual](https://github.com/Erechtheus/geolocation/blob/master/TrainIndividualModels.py) script. Pretrained models are available in HDF5 format [here](https://drive.google.com/open?id=0B9uTfq0OyHAsREphWG9OdHptREU). Additionally, we require some information on model and preprocessor (e.g., tokenizer) which is provided [here](https://drive.google.com/open?id=0B9uTfq0OyHAsZHRacHF3NDVObXc). The evaluation of models is implemented [here](https://github.com/Erechtheus/geolocation/blob/master/EvaluateTweet.py).

## Docker image

### Old deprecated docker image (V1.0)
Alternatively we provide a docker container [here](https://drive.google.com/open?id=0B9uTfq0OyHAsRDd1ZU9ldmxhTFE), containing processed data (e.g., tokenizers), pretrained models, evaluation data, and scripts. Extract, load, and connect to the container using:
```bash
unlzma geolocation.docker.lzma
docker load --input geolocation.docker
docker run -it geolocation:v1 bash
```

Evaluate performance by:
```bash
python3 /root/code/EvaluateTweet.py
python3 /root/code/EvaluateUser.py
```

### New docker image (V2.1)
We also provide a [docker image](https://drive.google.com/open?id=17aeTaCHcsW4vX6_RqD_-6NDQmlSG6uWk) of our code using functional API and a REST Service
```bash
unlzma geolocationV2.tar.lzma
docker load --input geolocationV2.tar
docker run -d -p   5000:5000 --network host  geoloc:latest
```

Access the simple text model using the [URL](http://127.0.0.1:5000/predictText?text=Montmartre%20is%20truly%20beautiful) and returns

```json
{
    "query": "Montmartre is truly beautiful",
    "results": [
        {
            "city": "paris-a875-fr",
            "lat": 48.857779087136095,
            "lon": 2.3539118329464914,
            "score": 0.43245163559913635
        },
        {
            "city": "boulogne billancourt-a892-fr",
            "lat": 48.82956285864007,
            "lon": 2.2603947479966044,
            "score": 0.045727577060461044
        },
        {
            "city": "saint denis-a893-fr",
            "lat": 48.947253923722585,
            "lon": 2.4314893304822607,
            "score": 0.0368279293179512
        },
        {
            "city": "creteil-a894-fr",
            "lat": 48.80814304627673,
            "lon": 2.5156099666530327,
            "score": 0.01906118169426918
        },
        {
            "city": "argenteuil-a895-fr",
            "lat": 48.97509961545753,
            "lon": 2.1906891017164387,
            "score": 0.01858099363744259
        }
    ]
}
```


## Example usage for short text:
The code below briefly describes how to use our neural network, trained on text only. For other examples (e.g., using Twitter text and metadata), we refer to the examples in the two evaluation scripts

```python
from keras.models import load_model
import pickle
from keras.preprocessing.sequence import pad_sequences
import numpy as np

#Load Model
textBranch = load_model('data/w-nut-latest/models/textBranchNorm.h5')

#Load tokenizers, and mapping
file = open("data/w-nut-latest/binaries/processors.obj",'rb')
descriptionTokenizer, domainEncoder, tldEncoder, locationTokenizer, sourceEncoder, textTokenizer, nameTokenizer, timeZoneTokenizer, utcEncoder, langEncoder, timeEncoder, placeMedian, classes, colnames = pickle.load(file)

#Load properties from model
file = open("data/w-nut-latest/binaries/vars.obj",'rb')
MAX_DESC_SEQUENCE_LENGTH, MAX_LOC_SEQUENCE_LENGTH, MAX_TEXT_SEQUENCE_LENGTH, MAX_NAME_SEQUENCE_LENGTH, MAX_TZ_SEQUENCE_LENGTH = pickle.load(file)
#Predict text (e.g., 'Montmartre is truly beautiful')
testTexts=[];
testTexts.append("Montmartre is truly beautiful")

textSequences = textTokenizer.texts_to_sequences(testTexts)
textSequences = np.asarray(textSequences)
textSequences = pad_sequences(textSequences, maxlen=MAX_TEXT_SEQUENCE_LENGTH)

predict = textBranch.predict(textSequences)

#Print the top 5
for index in reversed(predict.argsort()[0][-5:]):
    print("%s with score=%.3f" % (colnames[index], float(predict[0][index])) )
```

### The output is:
	paris-a875-fr with score=0.275
	city of london-enggla-gb with score=0.079
	boulogne billancourt-a892-fr with score=0.032
	saint denis-a893-fr with score=0.024
	meaux-a877-fr with score=0.015
