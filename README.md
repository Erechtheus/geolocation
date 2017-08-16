# geolocation
Geolocation prediction for a given Tweet, or a short text. The system trains a neural net, as described in

	Philippe Thomas and Leonhard Hennig (2017), "Twitter Geolocation Prediction using Neural Networks." In Proceedings of GSCL

# Usage

## Train and apply models
To train models, training data (tweets and gold labels) needs to be retrieved. As Tweets can not be shared directly, we refer to the [WNUT'16 workshop page](http://noisy-text.github.io/2016/geo-shared-task.html) for further information.

After retrieving the training files, the [preprocess](https://github.com/Erechtheus/geolocation/blob/master/Preprocess.py) script converts tweets into the desired representation to train a neural network. Models can be trained from scratch using the [trainindividual](https://github.com/Erechtheus/geolocation/blob/master/TrainIndividualModels.py) script. Pretrained models are available in HDF5 format [here](https://drive.google.com/open?id=0B9uTfq0OyHAsREphWG9OdHptREU). Additionally, we require some information on model and preprocessor (e.g., tokenizer) which is provided [here](https://drive.google.com/open?id=0B9uTfq0OyHAsZHRacHF3NDVObXc). The evaluation of models is implemented [here](https://github.com/Erechtheus/geolocation/blob/master/EvaluateTweet.py).

## Docker image 
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

