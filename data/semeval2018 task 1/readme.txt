*******************************************
SemEval-2018 Task 1: Affect in Tweets
*******************************************

The format of the data for V-reg (valence) is the following:

id[tab]tweet[tab]valence[tab]score

The scores are real values between 0 and 1: 0 indicates the lowest amount of valence, 1 indicates the highest amount of valence. 

Ids are our internal IDs generated separately for the training, development, and test data. The tweets in the training data come from the 2017 EI-reg training dataset. In 2017, those tweets were annotated for one of four emotions (joy, sadness, fear, or anger). Now (in this release), they are annotated for valence. FWIW, the old IDs (from 2017) for these tweets can be found as the last portion of the new ID. For example, the new ID '2018-en-valence-train-1-2017-30100' indicates that this tweet was used in the 2017 EI-reg training set with ID '30100'.



