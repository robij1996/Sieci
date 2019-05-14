
import numpy as np
import pandas as pd
from function import age, clas, sex, sib, parch, cabin, prepareData, fare, classifier, quadratic
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")


    

pd.set_option('display.expand_frame_repr', False)
# train, test = prepareData(train,test)
quadratic(train,test)


# classifier(train,test)










