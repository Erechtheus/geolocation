# geolocation
Geolocation prediction for a given Tweet, or a short text. The system trains a neural net, as described in

	Philippe Thomas and Leonhard Hennig (2017), "Twitter Geolocation Prediction using Neural Networks." In Proceedings of GSCL

# Performance
This section briefly provides some information about the performance of our method. We removed the original model and only provide information about the new model.

|Model   	|   Acc	| Median  	| Mean  	|  Acc 	| Median 	| Mean |
|---	|---	|---	|---	|---	|---	|---	|
|   Location	| 0.364| 208.0| 4525.7| 0.447| 42.0| 3811.1 |
|  Text 	    | 0.2| 1797.8| 4083.3 | 0.336| 199.2| 2291.7 |
|  Description 	| 0.096| 3152.5| 5805.0 |0.119| 2794.7| 5480.7 |
|  User-name 	| 0.059| 3838.7| 5934.8 |0.059| 3944.8| 6076.7 |
|  Timezone     | 0.057| 5268.0| 5528.7 | 0.061| 5471.7| 5463.2|
|  User-lang 	| 0.063| 6049.3| 7339.9 | 0.046| 9050.0| 8573.3 |
|  Links 	    | 0.033| 7605.3| 6980.3 | 0.045| 6733.1| 6576.7 |
|  UTC 	        | 0.047| 7709.5| 6850.0 | 0.051| 3888.6| 6423.1|
|  Source 	    | 0.045| 7998.1| 7521.5 | 0.045| 6982.7| 6981.1 |
|  Tweet-time 	| 0.028| 8398.1| 7668.8 | 0.024| 11720.6| 10241.8|
|  Full-fixed 	| **0.433**| **47.0**| **1152.4**  |**0.533**| **13.8**| **769.7** |
|  Baseline 	| 0.028  	| 11,723.0  | 10,264.3  | 0.024  	| 11,771.5  | 10,584.4  |


# Usage

## Download 
Source code from this repository has been published here (https://github.com/Erechtheus/geolocation/releases). 
- [Version 2.2](https://github.com/Erechtheus/geolocation/releases/tag/V2.2)   uses keras functional API (instead of Keras sequential API). This code runs with Keras Version 2, whereas the original [release](https://github.com/Erechtheus/geolocation/releases/tag/V1.0) worked with Keras Version 1. Also has some minor improvement regarding preprocessing and has a REST-API.


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

We provide a [docker image](https://drive.google.com/file/d/1mOJ7Hl12GSyG8LcV8cOcs-cpWwWu2Kq2/view?usp=sharing) of our code using functional API and a REST Service
```bash
unlzma geolocationV2.tar.lzma
docker load --input geolocationV2.tar
docker run -d -p   5000:5000 --network host  geoloc:latest
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
            "score": 0.2016402930021286
        },
        {
            "city": "city of london-enggla-gb",
            "lat": 51.50090096289424,
            "lon": -0.09162320754762229,
            "score": 0.08580838143825531
        },
        {
            "city": "boulogne billancourt-a892-fr",
            "lat": 48.82956285864007,
            "lon": 2.2603947479966044,
            "score": 0.030901918187737465
        },
        {
            "city": "manhattan-ny061-us",
            "lat": 40.760731485273375,
            "lon": -73.96936825522386,
            "score": 0.018226830288767815
        },
        {
            "city": "dublin-l33-ie",
            "lat": 53.37821923430317,
            "lon": -6.37129742197171,
            "score": 0.01762479543685913
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

#### The output is:
	paris-a875-fr with score=0.202
    city of london-enggla-gb with score=0.086
    boulogne billancourt-a892-fr with score=0.031
    manhattan-ny061-us with score=0.018
    dublin-l33-ie with score=0.018

## Train and apply models
To train models, training data (tweets and gold labels) needs to be retrieved. As Tweets can not be shared directly, we refer to the [WNUT'16 workshop page](http://noisy-text.github.io/2016/geo-shared-task.html) for further information.

After retrieving the training files, the [preprocess](https://github.com/Erechtheus/geolocation/blob/master/Preprocess.py) script converts tweets into the desired representation to train a neural network. Models can be trained from scratch using the [trainindividual](https://github.com/Erechtheus/geolocation/blob/master/TrainIndividualModels.py) script.
Pretrained models and preprocessors (e.g., used tokenizer)  are available [here](https://drive.google.com/file/d/11S76MWFT14vcraJ2V7skGIaKLpyQpks8/view?usp=sharing).
The evaluation of models is implemented [here](https://github.com/Erechtheus/geolocation/blob/master/EvaluateTweet.py).


# Possible improvements
 - Transformer models
 - LSTM representation via Keras generators to save memory
 - REST API with twitter JSON object as input
 - How's the performance for the full network when we only feed partial info? (E.g. only text, timezone, ...)
 - Incorporate user-graph for prediction (e.g. using neural structure learning)
 - Character CNN (memory consumption pretty high in my implementation, needs generators)
 - Use image data

 
# Tested improvements
 - FastText as embedding method -> Performance for text-model is below our current methods. But, we did not use a fast-text model explicitly learned on social-media data
