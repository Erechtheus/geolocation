# geolocation
Geolocation prediction for a given Tweet, or a short text. The system trains a neural net, as described in ...

# Usage


## Train models
To train models, training data (tweets and gold labels) needs to be retrieved. As Tweets can not be shared directly, we refer to the [WNUT'16 workshop page](http://noisy-text.github.io/2016/geo-shared-task.html) for further information.

After retrieving the training files, the [preprocess](https://github.com/Erechtheus/geolocation/blob/master/Preprocess.py) script converts tweets into the desired representation to train a neural network. Models can be trained from scratch using the [trainindividual](https://github.com/Erechtheus/geolocation/blob/master/TrainIndividualModels.py) script. Pretrained models are available in HDF5 format [here](https://drive.google.com/file/d/0B9uTfq0OyHAseWU1Z3pGYjdELTg/view?usp=sharing). Additionally, we require some information on model and preprocessor (e.g., tokenizer) which is provided [here](). The evaluation of models is implemented [here](https://github.com/Erechtheus/geolocation/blob/master/EvaluateTweet.py).

## Example usage for short text

