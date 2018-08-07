# Naive Bayes Classifier

Written for COMP30027 Machine Learning. This classifier takes CSV files with nominal attribute values and a class on the rightmost column for each instance. It uses laplace smoothing when dealing with unseen events. 

It can be trained with supervised labelled data and it will evaluate its performance based on the same training data. It can also perform unsupervised learning and then evaluate its performance based on the provided labels.

## Usage 

Change the CSV_FILE variable to one of the provided CSVs in assignment.py and run assignment.py. You can also use your own CSV file given that it only has nominal attribute values and is labelled with classes in the right most column. 
