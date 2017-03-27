# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid') 
#matplotlib inline


cmap = cm.get_cmap('Spectral') # Colour map (there are many others)

# machine learning

from sklearn.ensemble import RandomForestClassifier

# load training and testing data as a dataframe
train_df = pd.read_csv("./download/train.csv", dtype={"Age": np.float64}, )
test_df = pd.read_csv("./download/test.csv", dtype={"Age": np.float64}, )


train_df.head()
train_df.info()


# training set information

#Data columns (total 12 columns):
#PassengerId    891 non-null int64
#Survived       891 non-null int64
#Pclass         891 non-null int64
#Name           891 non-null object
#Sex            891 non-null object
#Age            714 non-null float64
#SibSp          891 non-null int64
#Parch          891 non-null int64
#Ticket         891 non-null object
#Fare           891 non-null float64
#Cabin          204 non-null object
#Embarked       889 non-null object
#dtypes: float64(2), int64(5), object(5)
#memory usage: 90.5+ KB


# testing sets information
test_df.info()
#<class 'pandas.core.frame.DataFrame'>
#Int64Index: 418 entries, 0 to 417
#Data columns (total 11 columns):
#PassengerId    418 non-null int64
#Pclass         418 non-null int64
#Name           418 non-null object
#Sex            418 non-null object
#Age            332 non-null float64
#SibSp          418 non-null int64
#Parch          418 non-null int64
#Ticket         418 non-null object
#Fare           417 non-null float64
#Cabin          91 non-null object
#Embarked       418 non-null object
#dtypes: float64(2), int64(4), object(5)
#


#Categorical: Survived, Sex, and Embarked. Ordinal: Pclass.
#Continous: Age, Fare. Discrete: SibSp, Parch.
#Ticket is a mix of numeric and alphanumeric data types. Cabin is alphanumeric.

################################
#
# Feature engineering and data visualization
#
################################

# Drop Ticket feature; it contains high ratio of duplicates (22%) and there may not be a correlation between Ticket and survival.
# Drop Cabin feature;  highly incomplete or contains many null values both in training and test dataset.
#PassengerId may be dropped from training dataset as it does not contribute to survival

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)


# combine training and testing datasets to run certain operations on both datasets together. 
combine = [train_df, test_df]

print "After combine", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape

 

###########################################################    
## processing 'Title', and mapping it as ordinal 
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])



for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()




title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)




train_df.head()


################
# Embarked ports
# Map it as ordinal

freq_port = train_df.Embarked.dropna().mode()[0]
freq_port
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_df.head()


#############
# Fare
# group it into 4 bands


test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

#grid = sns.FacetGrid(train_df, row = 'Pclass', size=2.2, aspect=1.6)
#grid.map(plt.hist, 'Fare',  alpha=.5)
#grid.add_legend()


fare_not_survived = train_df["Fare"][train_df["Survived"] == 0]
fare_survived = train_df["Fare"][train_df["Survived"] == 1]
# get average and std for fare of survived/not survived passengers
avgerage_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])
std_fare = DataFrame([fare_not_survived.std(), fare_survived.std()])
# plot
#fare_ax = train_df['Fare'].plot(kind='hist', figsize=(15,3),bins=100, xlim=(0,50), title='Fare histogram')
#fare_ax = train_df['Fare'].plot(kind='hist', figsize=(15,5),bins=100, xlim=(0,50), title='Fare histogram', color ='green')

ax_not_survived = plt.hist(fare_not_survived, normed=1,bins= 20, linestyle ='solid', hatch='//', range= [0,100], alpha=0.5, label='Not Survived')
ax_survived = plt.hist(fare_survived, normed=1, bins= 20, linestyle='dashed', hatch='\\', range= [0,100], alpha=0.5, label='Survived')


plt.legend(loc='upper right')
plt.yscale('log')

plt.xlabel('Fare')
plt.ylabel('Normed frequency')
#plt.show()
#plt.gcf().clear()


train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)


for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 5, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 5) & (dataset['Fare'] <= 10), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 10) & (dataset['Fare'] <= 80), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 80, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
    

#######################################################
##Sex

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)




#######################################################

###Age


#age_not_survived = train_df["Age"][train_df["Survived"] == 0]
#age_survived = train_df["Age"][train_df["Survived"] == 1]

#ax_not_survived = plt.hist(age_not_survived, normed=1,bins= 10, linestyle ='solid', hatch='//', range= [0,80], alpha=0.5, label='Not Survived')
#ax_survived = plt.hist(age_survived, normed=1, bins= 20, linestyle='dashed', hatch='\\', range= [0,80], alpha=0.5, label='Survived')

#plt.legend(loc='upper right')
#plt.yscale('log')

#plt.xlabel('Age')
#plt.ylabel('Normed frequency')
#plt.show()
#plt.gcf().clear()


# calculate the medium age for each class. And assign the medium value to the passengers with null value age.... according to their classes.
guess_ages = np.zeros((2,3))
guess_ages


for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
  
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)
            age_guess = guess_df.median()
            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),'Age'] =  guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)




for dataset in combine:    
    dataset.loc[ dataset['Age'] <=13 , 'Age'] = 0
    dataset.loc[(dataset['Age'] > 13) & (dataset['Age'] <= 26), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 47), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 47, 'Age'] = 4





rate = train_df[['Age', 'Survived']].groupby(['Age'], as_index=False).mean().sort_values(by='Age', ascending=True)
print 'age-survial rate', rate


train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
train_df.head()


train_df.info()

###########
# age*pclass
#for dataset in combine:
#    dataset['Age*Class'] = dataset.Age * dataset.Pclass

#train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)

#######################################################

### Family

for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

familysize_survival_rate = train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#print familysize_survival_rate

for dataset in combine:
   dataset.loc[dataset['FamilySize'] > 4, 'FamilySize'] = 5

train_df = train_df.drop(['Parch', 'SibSp'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp'], axis=1)


#===================
#train_df['SibSp'].loc[train_df['SibSp'] > 3] = 4
#test_df['SibSp'].loc[test_df['SibSp'] > 3] = 4
#train_df['Parch'].loc[train_df['Parch'] > 3] = 4
#test_df['Parch'].loc[test_df['Parch'] > 3] = 4



#for dataset in combine:
#    dataset['IsAlone'] = 0
#    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

#train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

#train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
#test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]



#############################################

# drop unnecessary columns, these columns won't be useful in analysis and prediction
train_df = train_df.drop(['PassengerId','Name'], axis=1)
test_df = test_df.drop(['Name'], axis=1)



######################

# define training and testing sets
X_train = train_df.drop("Survived",axis=1)
Y_train = train_df["Survived"]
X_test = test_df.drop("PassengerId",axis=1).copy()

X_test.info()
X_train.info()

################################


# Random Forests
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
print random_forest.score(X_train, Y_train)


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
})
submission.to_csv('train.csv', index=False)
