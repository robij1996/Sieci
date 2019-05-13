import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import warnings


def clas(train, test):
    sns.barplot(x="Pclass", y="Survived", data=train)
    plt.show()

def age(train, test):
    train["Age"] = train["Age"].fillna(-0.5)
    test["Age"] = test["Age"].fillna(-0.5)
    bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
    labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)
    test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)

    return train, test
    # sns.barplot(x="AgeGroup", y="Survived", data=train)
    # plt.show()

def sex(train,test):
    sns.barplot(x="Sex", y="Survived", data=train)
    plt.show()

def sib(train, test):
    sns.barplot(x="SibSp", y="Survived", data=train)
    plt.show()

def parch(train, test):
    sns.barplot(x="Parch", y="Survived", data=train)
    plt.show()

def cabin(train, test):
    train["CabinBool"] = (train["Cabin"].notnull().astype('int'))
    test["CabinBool"] = (test["Cabin"].notnull().astype('int'))
    sns.barplot(x="CabinBool", y="Survived", data=train)
    plt.show()

def fare(train, test):
    bins = [-1, 7.9, 14.4, 31, np.inf]
    labels = ['Cheap', 'Average', 'Expensive', 'VIP']
    train['Fare'] = pd.cut(train["Fare"], bins, labels = labels)
    test['Fare'] = pd.cut(test["Fare"], bins, labels = labels)
    return train, test
    # sns.barplot(x="Fare", y="Survived", data=train)
    # plt.show()


def cleanData(train, test):
    for data in range(len(train)):
        if(np.isnan(train['Age'][data])):
            train['Age'][data] = np.random.randint(70)+1

    for data in range(len(test)):
        if(np.isnan(test['Age'][data])):
            test['Age'][data] = np.random.randint(70)+1
    
    train, test = age(train, test)
    train, test = fare(train,test)
    drop_elements = ['PassengerId', 'Name', 'Ticket', 'Embarked', 'Cabin', 'Age']
    train = train.drop(drop_elements, axis = 1)
    test  = test.drop(drop_elements, axis = 1)

    return train, test

def prepareData(train, test):

    train, test = cleanData(train, test)

    sex_mapping = {"male": 0, "female": 1}
    train['Sex'] = train['Sex'].map(sex_mapping)
    test['Sex'] = test['Sex'].map(sex_mapping)

    age_title_mapping = {'Baby': 0, 'Child': 1, 'Teenager':2, 'Student':3, 'Young Adult':4, 'Adult':5, 'Senior':6}
    train['AgeGroup'] = train['AgeGroup'].map(age_title_mapping).astype(int)
    test['AgeGroup'] = test['AgeGroup'].map(age_title_mapping).astype(int)

    fare_mapping = {'Cheap': 0, 'Average': 1, 'Expensive': 2, 'VIP': 3}
    train['Fare'] = train['Fare'].map(fare_mapping)
    test['Fare'] = test['Fare'].map(fare_mapping)

    return train, test

    

   





