# Machine Learning Project 2 - Text Sentiment Classification of Tweets

This project contains the code used for the text sentiment analysis task over tweets organized during the Machine Learning course at EPFL.
The task is to build a model to predict if a tweet used to contain a positive :) or negative :( smiley. 

## Dataset overview

The data is composed of 2500000 train tweets and 10000 test tweets. The file we used are:

 -  train_pos.txt and train_neg.txt - a small set of training tweets for each of the two classes.
 -  train_pos_full.txt and train_neg_full.txt - a complete set of training tweets for each of the two classes, about 1M tweets per class.
 -  test_data.txt - the test set, that is the tweets for which you have to predict the sentiment label.

## Project overview

The project is organized in several files in order to guarantee modularity and have a clear structure: 

 - `run.py` is a simple script to generate our best submission by using a pre-trained model.
 - `cleaner.py` contains classical utlities that were used to clean and preprocess text data (for example, removing punctuation and spaces, filtering stopwords, lemmatization etc.). 
 - `helpers.py` has a set of functions that are used across the whole project for different goals, like aggregating data or generating tokens from text.
 - `cross_validation.py` contains some of the functions that were used to cross validate our models for local testing and tuning of the hyper-parameters in order to get the best model.
 - `lstm_200.h5` is the pre-trained model saved that produced the best reproducible result in terms of accuracy. 
 - `Data_Exploration.ipynb` is a jupyter notebook that shows some initial and preliminar analysis of the data. 
 - `Models-Trainer.ipynb` is the notebook we used to train models offered by the Scikit-Learn library. In particular, we focused on the following classifiers: Logistic Regression, Random Forest, Support Vector Machine and Naive Bayes.  
 - `CNN-Trainer.ipynb` is the notebook created for testing our Neural Networks with Convolutional layers by using TensorFlow and Keras.
 - `RNN-Trainer.ipynb` is the notebook created for testing our Neural Networks with LSTM by using TensorFlow and Keras.
 - `scripts` contains some files that can generate a vocabulary of the training set words, the corresponding coocurrence matrix and the word embeddings that were used for Scikit-learn models.
 - `tools` has already some of the files generated with the utilities in the script folder such as the vocabulary and the embeddings.
 

## Getting Started

These instructions will provide you all the information to get the best result we achieved on your local machine, as described above.

### Dataset preparation 

It is necessary to include a `data` folder in the project with the data required for training and testing. In particular, you need `train_pos_full.txt` and `train_neg_full.txt` for training and `test_data.txt` for testing.
  
  
### Environment set-up

We recommend you to use an Anaconda environment with Python 3.6 for installing the packages required. The main packages we used to run the code:

- numpy 1.15.4
- pandas 0.23.4
- Tensorflow 1.12.0
- Scikit-Learn 0.20.1
- NLTK 3.3.0
- TextBlob 0.15.2

All the information to install the dependencies can be easily found on the web. 

### Creating the prediction

Once the dataset have been set-up, just execute `run.py` to get our best model which reproduce 0.843 accuracy on the test set:

```
python run.py
```

You will see some output on the screen. Once "Done" appears, you will be able to see that a `best_submission.csv` file has been generated. This file contains the predictions with the best model and parameters we could find and replicates exactly our best submission on the CrowdAI competition.

## Notes

Unfortunately, our best submission for the competition organized by the EPFL was higher than the one we provide here. In particular, we got 0.864 of accuracy but we can't consistenly reproduce the result in all the machines. One possible cause is related to the fact we used a Google Cloud Virtual Machine to train our models and we had to deal with different settings and packages versions respect to our configurations. Another issue to consider is that several users have trouble to reproduce exactly the same results after saving the model with Keras. This seems to be a common issue on the forums and we are not the only ones that experienced this with different models.

## Authors 

 - Marshall Cooper
 - Andrea Piccione
 - Divij Pherwani

## Acknowledgments

The entire project used some utlities functions provided during the course that can be found on https://github.com/epfml/ML_course
