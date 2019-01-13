# AffectInTweets
Finding the intensity of emotions in tweets (SemEval 2018 Task 1 Participant)

## Introduction

Existing emotion analysis tools generally classify a given text into predefined emotion categories (positive, negative, or angry, cheerful, sad, etc.).
However, knowing the intensity of emotion expressed in a given text would be useful for many applications such as brand and product perception tracking.
In this study, a system was developed to measure the intensity of emotion in tweets by using shared datasets in the workshop named  Affects in Tweets took place in SemEval 2018 - Task 1.

In the data set there are anonymous tweets and affective intensities between 0 and 1 determined by annotators for each emotion (anger, fear, fun and sadness).
In this study, a model based on bidirectional LSTM networks, which provides highly successful results in natural language processing tasks is examined for finding emotion intensities of tweets.
The results show that there is a significant increase in the performance when LSTM based model initialized with a pretrained model which is trained on a sentiment140 dataset and lexicon based features extracted from emoton lexicons are also utilized.

A simple user interface is provided to try the models. Follow the instructions below to run web based application.

*A screenshot from the application*

![screenshot](../master/resources/assets/Capture1.PNG?raw=true)

![screenshot](../master/resources/assets/Capture2.PNG?raw=true)

# Experimental Results

| Method | Joy |  | Öfke |  | Üzüntü |  | Korku |  | Valans |  |
|--------------------------------------------------------|-------------------------|--------------------|-------------------------|--------------------|-------------------------|--------------------|-------------------------|--------------------|-------------------------|--------------------|
|  | Classification Accuracy | Pearson Corelation | Classification Accuracy | Pearson Corelation | Classification Accuracy | Pearson Corelation | Classification Accuracy | Pearson Corelation | Classification Accuracy | Pearson Corelation |
| Bidirectional LSTM | 41.3 | 0.49 | 42.01 | 0.35 | 40.8 | 0.47 | 60.41 | 0.49 | 31.84 | 0.32 |
| Bidirectional LSTM + Lexicon Features | 43.1 | 0.54 | 45.61 | 0.43 | 47.6 | 0.47 | 63.75 | 0.55 | 36.3 | 0.51 |
| Bidirectional LSTM with pretraining | 42.06 | 0.62 | 46.9 | 0.48 | 49.87 | 0.63 | 64.26 | 0.58 | 38.53 | 0.68 |
| Bidirectional LSTM with pretraining + Lexicon Features | 43.1 | 0.6 | 46.9 | 0.5 | 52.64 | 0.64 | 66.06 | 0.55 | 40.31 | 0.71 |

## System Requirements
You should run the system on a machine where ptyhon-2.7 has been installed.
The following packages are required to run the system.
```
numpy=1.13.1
gensim==3.0.0
scipy==0.19.1
nltk==3.2.4
tensorflow==1.2.1
keras==2.0.6
tornado==4.5.1
```

## Running the program
The program has a web based user interface. 
You can run the web application just running the following command:
```
python src/web.py
```

The system will load required models to the memory and it will be ready when you see the following output in the console:
```
Loading emotion models...
Done.
```

Now you can reach the user interface through a web browser from following url:
```
http://localhost:8891
```

To run the program from command line run the following code:
```
python src/affect_predictor.py "tweet to be analyzed"
```

You can also use the program within python code using programming API:
```python
from affect_predictor import AffectPredictor
AffectPredictor().predict("tweet to be analyzed")
```

If you want to train models from scratch, you can use the following command:
```
python src/affect_model_trainer.py
```
