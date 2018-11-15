# TitanicDatasets

import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
from sklearn import datasets, svm, cross_validation, tree, preprocessing, metrics
import sklearn.ensemble as ske
titanic_df = pd.read_excel('titanic3.xls', 'titanic3', index_col=None, na_values=['NA']) 
titanic_df.head(5) #reading top  values from the datasets
titanic_df['survived'].mean() #finding the mean of output attributes
titanic_df.groupby('pclass').mean()
class_sex_grouping = titanic_df.groupby(['pclass','sex']).mean()
class_sex_grouping
class_sex_grouping['survived'].plot.bar() #Visulisation with the help of bar graph
group_by_age = pd.cut(titanic_df["age"], np.arange(0, 90, 10))
age_grouping = titanic_df.groupby(group_by_age).mean()
age_grouping['survived'].plot.bar()
titanic_df.count() #counting of values from all the attribute
titanic_df = titanic_df.drop(['body','cabin','boat'], axis=1) #droping irrelevent attributes from the datasets
titanic_df["home.dest"] = titanic_df["home.dest"].fillna("NA")  #Handling the missing values
titanic_df = titanic_df.dropna()
titanic_df.count()

def preprocess_titanic_df(df): #preprocessing of our titanic datasets
    processed_df = df.copy()
    le = preprocessing.LabelEncoder()
    processed_df.sex = le.fit_transform(processed_df.sex)
    processed_df.embarked = le.fit_transform(processed_df.embarked)
    processed_df = processed_df.drop(['name','ticket','home.dest'],axis=1)
    return processed_df
    

processed_df = preprocess_titanic_df(titanic_df)
X = processed_df.drop(['survived'], axis=1).values
y = processed_df['survived'].values

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2) #spliting the datasets in training and test dataset 

clf_dt = tree.DecisionTreeClassifier(max_depth=10)#initialising the Decision tree classifier with depth of 10
clf_dt.fit (X_train, y_train) #making our classifer to learn the training datsets
clf_dt.score (X_test, y_test) #checking of our classifer on test datsets

shuffle_validator = cross_validation.ShuffleSplit(len(X), n_iter=20, test_size=0.2, random_state=0)
def test_classifier(clf):  #using cross validation with no of iteration is=20
    scores = cross_validation.cross_val_score(clf, X, y, cv=shuffle_validator)
    print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std()))
    
test_classifier(clf_dt) 

clf_rf = ske.RandomForestClassifier(n_estimators=50) #Using Random_Forest classifier
test_classifier(clf_rf)

clf_gb = ske.GradientBoostingClassifier(n_estimators=50) #Using Boosting Algorithm for better results
test_classifier(clf_gb)

eclf = ske.VotingClassifier([('dt', clf_dt), ('rf', clf_rf), ('gb', clf_gb)])
test_classifier(eclf)
