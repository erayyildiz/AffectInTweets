# AffectInTweets
Find the intensity of affects in tweets (SemEval 2018 Task 1 Participant)

# System Requirements
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

# Running the program
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