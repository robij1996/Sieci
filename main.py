
import numpy as np
import pandas as pd
from function import age, clas, sex, sib, parch, cabin, prepareData, fare
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")


pd.set_option('display.expand_frame_repr', False)
train, test = prepareData(train,test)

features = ['Pclass',  'Sex',  'SibSp',  'Parch',  'Fare',  'AgeGroup']
y = train['Survived']
X = train[features]

model = GradientBoostingClassifier()
model.fit(X, y)

print("The predictions are")
print(model.predict(test.head(10)))










