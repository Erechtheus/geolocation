# geolocation
Geolocation prediction for a given Tweet, or a short text. The system trains a neural net, as described in

	Philippe Thomas and Leonhard Hennig (2017), "Twitter Geolocation Prediction using Neural Networks." In Proceedings of GSCL

# Performance

## Old (published) model
|Model   	|   Acc	| Median  	| Mean  	|  Acc 	| Median 	| Mean |
|---	|---	|---	|---	|---	|---	|---	|
|   Location	| 0.361  	| 205.6  	| 4,538.0 	| 0.445  	| 43.9  	| 3,831.7 	|
|  Text 	    | 0.195 	| 2,190.6  	| 4,472.9	| 0.321  	| 263.8 	| 2,570.9	|
|  Description 	| 0.087 	| 3,817.2 	| 6,060.2 	| 0.098 	| 3,296.9  	| 5,880.0 	|
|  User-name 	| 0.057 	| 3,849.0 	| 5,930.1	| 0.059  	| 4,140.4  	| 6,107.6 	|
|  Timezone     | 0.058     | 5,268.0   | 5,530.1   | 0.061     | 5,470.5   | 5,465.5   |
|  User-lang 	| 0.061  	| 6,465.1  	| 7,310.2 	| 0.047  	| 8,903.7 	| 8,525.1 	|
|  Links 	    | 0.032  	| 7,601.7  	| 6,980.5  	| 0.045  	| 6,687.4  	| 6,546.8  	|
|  UTC 	        | 0.046  	| 7,698.1  	| 6,849.0  	| 0.051  	| 3,883.4  	| 6,422.6  	|
|  Source 	    | 0.045  	| 8,005.0  	| 7,516.8  	| 0.045  	| 6,926.3  	| 6,923.5  	|
|  Tweet-time 	| 0.028  	| 8,867.6  	| 8,464.9  	| 0.024  	| 11,720.6  | 10,363.2  |
|  Full-scratch | 0.417  	|   59.0	| 1,616.4  	| 0.513  	| 17.8  	| 1,023.9  	|
|  Full-fixed 	| **0.430**  	|   **47.6**	| **1,179.4**  	| **0.530**  	| **14.9**  	| **838.5**  	|
|  Baseline 	| 0.028  	| 11,723.0  | 10,264.3  | 0.024  	| 11,771.5  | 10,584.4  |

## New model (Version 2.0)
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
- [Version 1.0](https://github.com/Erechtheus/geolocation/releases/tag/V1.0) refers to the original code, which has been presented in our GSCL paper. 
- [Version 2.2](https://github.com/Erechtheus/geolocation/releases/tag/V2.2)  (**highly** recommended) uses keras functional API (instead of Keras sequential API). This code runs with Keras Version 2, whereas the original release worked with Keras Version 1. Also has some minor improvement regarding preprocessing and has a REST-API.

## Train and apply models
To train models, training data (tweets and gold labels) needs to be retrieved. As Tweets can not be shared directly, we refer to the [WNUT'16 workshop page](http://noisy-text.github.io/2016/geo-shared-task.html) for further information.

After retrieving the training files, the [preprocess](https://github.com/Erechtheus/geolocation/blob/master/Preprocess.py) script converts tweets into the desired representation to train a neural network. Models can be trained from scratch using the [trainindividual](https://github.com/Erechtheus/geolocation/blob/master/TrainIndividualModels.py) script.
Pretrained models and preprocessors (e.g., used tokenizer)  are available [here](https://drive.google.com/file/d/11S76MWFT14vcraJ2V7skGIaKLpyQpks8/view?usp=sharing).
The evaluation of models is implemented [here](https://github.com/Erechtheus/geolocation/blob/master/EvaluateTweet.py).

##  Local installation
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

### New docker image with REST-API (V2.2); recomended
We also provide a [docker image](https://drive.google.com/file/d/1mOJ7Hl12GSyG8LcV8cOcs-cpWwWu2Kq2/view?usp=sharing) of our code using functional API and a REST Service
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
}```


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
	paris-a875-fr with score=0.202
    city of london-enggla-gb with score=0.086
    boulogne billancourt-a892-fr with score=0.031
    manhattan-ny061-us with score=0.018
    dublin-l33-ie with score=0.018

# Possible improvements
 - Character CNN
 - Transformer models
 - Use image data
 - LSTM representation via Keras generators to save memory
 - Incorporate user-graph for prediction (e.g. using neural structure learning)
 - How's the performance for the full network when we only feed partial info? (E.g. only text, timezone, ...)