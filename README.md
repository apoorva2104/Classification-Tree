# Classification-Tree
Running a Classification Tree

Decision trees are predictive models that allow for a data driven exploration of nonlinear relationships and interactions among many explanatory variables in predicting a response or target variable. When the response variable is categorical (two levels), the model is a called a classification tree. Explanatory variables can be either quantitative, categorical or both. Decision trees create segmentations or subgroups in the data, by applying a series of simple rules or criteria over and over again which choose variable constellations that best predict the response (i.e. target) variable.

 For this assignment purpose, i chose variables ‘HISPANIC’,’WHITE’,’BLACK’ only and a binary, categorical response variable (SMOKING).
 
All possible separations (categorical) or cut points (quantitative) are tested. For the present analyses, the entropy “goodness of split” criterion was used to grow the tree and a cost complexity algorithm was used for pruning the full tree into a final subtree.

The smoking score was the first variable to separate the sample into further subgroups based on smoking = Yes or No.

Prediction Score is 82%

Smokers with a score greater than 0.3328 were more likely to be Whites.

In [1]:

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics

In [2]:
os.chdir("F:/COURSERA COURSES/Machine Learning for Data Analysis/Week 1")

In [3]:
"""
Data Engineering and Analysis
"""
#Load the dataset

AH_data = pd.read_csv("tree_addhealth.csv")

In [4]:
AH_data.head()

Out[4]:
BIO_SEX	HISPANIC	WHITE	BLACK	NAMERICAN	ASIAN	age	TREG1	ALCEVR1	ALCPROBS1	…	ESTEEM1	VIOL1	PASSIST	DEVIANT1	SCHCONN1	GPA1	EXPEL1	FAMCONCT	PARACTV	PARPRES
0	2	0	0	1	0	0	NaN	0	1	2	…	47	4	0	5	NaN	NaN	0	24.3	8	15
1	2	0	0	1	0	0	19.427397	1	1	1	…	35	1	0	5	22	2.333333	0	23.3	9	15
2	1	0	1	0	0	0	NaN	0	0	0	…	45	0	0	1	30	2.250000	0	24.3	3	15
3	1	0	0	1	0	0	20.430137	1	0	0	…	47	4	1	4	19	2.000000	0	18.7	6	14
4	2	0	0	1	0	0	NaN	0	1	0	…	39	0	0	5	32	3.000000	0	20.0	9	6
5 rows × 25 columns

In [5]:

5 rows × 25 columns

In [5]:

data_clean = AH_data.dropna()

In [6]:
data_clean.dtypes

Out[6]:
BIO_SEX      float64
HISPANIC     float64
WHITE        float64
BLACK        float64
NAMERICAN    float64
ASIAN        float64
age          float64
TREG1        float64
ALCEVR1      float64
ALCPROBS1      int64
marever1       int64
cocever1       int64
inhever1       int64
cigavail     float64
DEP1         float64
ESTEEM1      float64
VIOL1        float64
PASSIST        int64
DEVIANT1     float64
SCHCONN1     float64
GPA1         float64
EXPEL1       float64
FAMCONCT     float64
PARACTV      float64
PARPRES      float64
dtype: object

In [7]:
data_clean.describe()

Out[7]:
BIO_SEX	HISPANIC	WHITE	BLACK	NAMERICAN	ASIAN	age	TREG1	ALCEVR1	ALCPROBS1	…	ESTEEM1	VIOL1	PASSIST	DEVIANT1	SCHCONN1	GPA1	EXPEL1	FAMCONCT	PARACTV	PARPRES
count	4575.000000	4575.000000	4575.000000	4575.000000	4575.000000	4575.000000	4575.000000	4575.000000	4575.000000	4575.000000	…	4575.000000	4575.000000	4575.000000	4575.000000	4575.000000	4575.000000	4575.000000	4575.000000	4575.000000	4575.000000
mean	1.521093	0.111038	0.683279	0.236066	0.036284	0.040437	16.493052	0.176393	0.527432	0.369180	…	40.952131	1.618579	0.102514	2.645027	28.360656	2.815647	0.040219	22.570557	6.290710	13.398033
std	0.499609	0.314214	0.465249	0.424709	0.187017	0.197004	1.552174	0.381196	0.499302	0.894947	…	5.381439	2.593230	0.303356	3.520554	5.156385	0.770167	0.196493	2.614754	3.360219	2.085837
min	1.000000	0.000000	0.000000	0.000000	0.000000	0.000000	12.676712	0.000000	0.000000	0.000000	…	18.000000	0.000000	0.000000	0.000000	6.000000	1.000000	0.000000	6.300000	0.000000	3.000000
25%	1.000000	0.000000	0.000000	0.000000	0.000000	0.000000	15.254795	0.000000	0.000000	0.000000	…	38.000000	0.000000	0.000000	0.000000	25.000000	2.250000	0.000000	21.700000	4.000000	12.000000
50%	2.000000	0.000000	1.000000	0.000000	0.000000	0.000000	16.509589	0.000000	1.000000	0.000000	…	40.000000	0.000000	0.000000	1.000000	29.000000	2.750000	0.000000	23.700000	6.000000	14.000000
75%	2.000000	0.000000	1.000000	0.000000	0.000000	0.000000	17.679452	0.000000	1.000000	0.000000	…	45.000000	2.000000	0.000000	4.000000	32.000000	3.500000	0.000000	24.300000	9.000000	15.000000
max	2.000000	1.000000	1.000000	1.000000	1.000000	1.000000	21.512329	1.000000	1.000000	6.000000	…	50.000000	19.000000	1.000000	27.000000	38.000000	4.000000	1.000000	25.000000	18.000000	15.000000
8 rows × 25 columns

In [72]:
"""
Modeling and Prediction
"""
#Split into training and testing sets

""" ORIGINAL DATA.... 
predictors = data_clean[['BIO_SEX','HISPANIC','WHITE','BLACK','NAMERICAN','ASIAN',
'age','ALCEVR1','ALCPROBS1','marever1','cocever1','inhever1','cigavail','DEP1',
'ESTEEM1','VIOL1','PASSIST','DEVIANT1','SCHCONN1','GPA1','EXPEL1','FAMCONCT','PARACTV',
'PARPRES']]
"""

# predictors = data_clean[['BIO_SEX','HISPANIC','WHITE','BLACK','NAMERICAN','ASIAN']]


predictors = data_clean[['DEVIANT1','HISPANIC','WHITE','BLACK']]
In [73]:
targets = data_clean.TREG1

pred_train, pred_test, tar_train, tar_test  =   train_test_split(predict)

In [74]:
pred_train.shape
Out[74]:
(2745, 3)
In [75]:
pred_test.shape
Out[75]:
(1830, 3)


