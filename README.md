# geolocation
Geolocation prediction for a given Tweet, or a short text. The system trains a neural net, as described in

	Philippe Thomas and Leonhard Hennig (2017), "Twitter Geolocation Prediction using Neural Networks." In Proceedings of GSCL

# Performance
This section briefly provides some information about the performance of our method. We removed the original model and only provide information about the new model.

|Model   	|   Acc	| Median  	| Mean  	|  Acc 	| Median 	| Mean |
|---	|---	|---	|---	|---	|---	|---	|
|   Location	| 0.366| 203.9  | 4514.1 | 0.448 | 41.7   | 3821.0 |
|  Text 	    | 0.201| 1834.8 | 4320.1 | 0.330 | 213.9  | 2441.7 |
|  Description 	| 0.096| 3335.7 | 5837.4 | 0.121 | 2800.3 | 5491.0 |
|  User-name 	| 0.060| 3852.3 | 5909.3 | 0.058 | 4154.9 | 6131.7 |
|  Timezone     | 0.057| 5280.1 | 5554.8 | 0.061 | 5489.9 | 5481.4 |
|  User-lang 	| 0.061| 6465.1 | 7306.9 | 0.047 | 8903.7 | 8523.1 |
|  Links 	    | 0.033| 7606.3 | 6984.0 | 0.045 | 6734.4 | 6571.7 |
|  UTC 	        | 0.045| 5387.4 | 5570.0 | 0.048 | 5365.8 | 5412.7 |
|  Source 	    | 0.045| 8026.4 | 7539.6 | 0.045 | 6903.5 | 6901.1 |
|  Tweet-time 	| 0.028| 8442.5 | 7694.4 | 0.024 | 11720.6| 10275.5 |
|  Full-fixed 	| **0.442**| **43.6**| **1151.0**  |**0.530**| **14.1**| **771.2** |
|  Baseline 	| 0.028  	| 11,723.0  | 10,264.3  | 0.024  	| 11,771.5  | 10,584.4  |


# Usage

## Download 
Source code from this repository has been published here (https://github.com/Erechtheus/geolocation/releases). 
- [Version 2.3](https://github.com/Erechtheus/geolocation/releases/tag/V2.3)   uses keras functional API (instead of Keras sequential API). This code runs with Keras Version 2, whereas the original [release](https://github.com/Erechtheus/geolocation/releases/tag/V1.0) worked with Keras Version 1. Also has some minor improvement regarding preprocessing and has a REST-API.


##  Local installation (python)
This section briefly explains the steps to download the source code, installs python dependencies in Anaconda, downloads the models and processors and performs text classification for one text example.
```bash
git clone https://github.com/Erechtheus/geolocation.git
cd geolocation/
conda create --name geoloc  --file requirements.txt
conda activate geoloc

#Download model and preprocessor https://drive.google.com/file/d/11S76MWFT14vcraJ2V7skGIaKLpyQpks8/view?usp=sharing

tar xfva modelsV2.tar.lzma
tar xfva processorsV2.tar.lzma

python predictText.py
```


## Local installation (docker image)

We provide a [docker image](https://drive.google.com/file/d/19AA3M6dZHK8gogC8qxmt2vVDXrvs98MR/view?usp=sharing) of our code using functional API and a REST Service
```bash
unlzma geolocationV2.tar.lzma
docker load --input geolocationV2.tar
docker run -d -p   5000:5000 --network host  geoloc:latest
```

Alternatively, you can download the model from [docker hub](https://hub.docker.com/r/erechtheus79/geolocation).

```bash
docker pull erechtheus79/geolocation
docker run -d -p   5000:5000 --network host  erechtheus79/geolocation
```

Access the simple text model using the [URL](http://127.0.0.1:5000/predictText?text=Montmartre%20is%20truly%20beautiful) and it returns

```json
{
    "query": "Montmartre is truly beautiful",
    "results": [
        {
            "city": "paris-a875-fr",
            "lat": 48.857779087136095,
            "lon": 2.3539118329464914,
            "score": 0.18563927710056305
        },
        {
            "city": "city of london-enggla-gb",
            "lat": 51.50090096289424,
            "lon": -0.09162320754762229,
            "score": 0.04953022673726082
        },
        {
            "city": "boulogne billancourt-a892-fr",
            "lat": 48.82956285864007,
            "lon": 2.2603947479966044,
            "score": 0.04159574210643768
        },
        {
            "city": "saint denis-a893-fr",
            "lat": 48.947253923722585,
            "lon": 2.4314893304822607,
            "score": 0.02842172235250473
        },
        {
            "city": "argenteuil-a895-fr",
            "lat": 48.97509961545753,
            "lon": 2.1906891017164387,
            "score": 0.021229125559329987
        }
    ]
}
```


## Example usage to predict location of a text snippet:
The code below briefly describes how to use our neural network, trained on text only. For other examples (e.g., using Twitter text and metadata), we refer to the examples in the two evaluation scripts

```python
from keras.models import load_model
import pickle
from keras.preprocessing.sequence import pad_sequences
import numpy as np

#Load Model
textBranch = load_model('data/models/textBranchNorm.h5')
from keras.models import load_model
import pickle
from keras.preprocessing.sequence import pad_sequences
import numpy as np

#Load Model
textBranch = load_model('data/models/textBranchNorm.h5')

#Load tokenizers, and mapping
file = open("data/binaries/processors.obj",'rb')
descriptionTokenizer, domainEncoder, tldEncoder, locationTokenizer, sourceEncoder, textTokenizer, nameTokenizer, timeZoneTokenizer, utcEncoder, langEncoder, timeEncoder, placeMedian, classes, colnames = pickle.load(file)

#Load properties from model
file = open("data/binaries/vars.obj",'rb')
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

#Load tokenizers, and mapping
file = open("data/binaries/processors.obj",'rb')
descriptionTokenizer, domainEncoder, tldEncoder, locationTokenizer, sourceEncoder, textTokenizer, nameTokenizer, timeZoneTokenizer, utcEncoder, langEncoder, timeEncoder, placeMedian, classes, colnames = pickle.load(file)

#Load properties from model
file = open("data/binaries/vars.obj",'rb')
MAX_DESC_SEQUENCE_LENGTH, MAX_LOC_SEQUENCE_LENGTH, MAX_TEXT_SEQUENCE_LENGTH, MAX_NAME_SEQUENCE_LENGTH, MAX_TZ_SEQUENCE_LENGTH = pickle.load(file)
#Predict text (e.g., 'Montmartre is truly beautiful')
testTexts=[]
testTexts.append("Montmartre is truly beautiful")

textSequences = textTokenizer.texts_to_sequences(testTexts)
textSequences = np.asarray(textSequences)
textSequences = pad_sequences(textSequences, maxlen=MAX_TEXT_SEQUENCE_LENGTH)

predict = textBranch.predict(textSequences)

#Print the top 5
for index in reversed(predict.argsort()[0][-5:]):
    print("%s with score=%.3f" % (colnames[index], float(predict[0][index])) )
```

#### The output is:
    paris-a875-fr with score=0.186
    city of london-enggla-gb with score=0.050
    boulogne billancourt-a892-fr with score=0.042
    saint denis-a893-fr with score=0.028
    argenteuil-a895-fr with score=0.021

## Train and apply models
To train models, training data (tweets and gold labels) needs to be retrieved. As Tweets can not be shared directly, we refer to the [WNUT'16 workshop page](http://noisy-text.github.io/2016/geo-shared-task.html) for further information.

After retrieving the training files, the [preprocess](https://github.com/Erechtheus/geolocation/blob/master/Preprocess.py) script converts tweets into the desired representation to train a neural network. Models can be trained from scratch using the [trainindividual](https://github.com/Erechtheus/geolocation/blob/master/TrainIndividualModels.py) script.
Pretrained models and preprocessors (e.g., used tokenizer)  are available [here](https://drive.google.com/open?id=1BA_Rj5FJ30nTzvfJvnhgx3k-bzC6Sn9D).
The evaluation of models is implemented [here](https://github.com/Erechtheus/geolocation/blob/master/EvaluateTweet.py).


# Possible improvements
 - Transformer models
 - LSTM representation via Keras generators to save memory
 - REST API with twitter JSON object as input
 - How's the performance for the full network when we only feed partial info? (E.g. only text, timezone, ...)
 - Incorporate user-graph for prediction (e.g. using neural structure learning)
 - Character CNN (memory consumption pretty high in my implementation, needs generators)
 - Use image data
 - Train a worldwide country-model? Clustering?
 
# Tested improvements
 - FastText as embedding method -> Performance for text-model is below our current methods. But, we did not use a fast-text model explicitly learned on social-media data
 - LSTM using recurrent dropout -> no improvement can be oberved (TrainIndividualModelsCNN.py)