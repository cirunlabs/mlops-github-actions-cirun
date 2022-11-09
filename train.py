# importing required libraries
# 1. for data manipulation
import numpy as np
import pandas as pd
# 2. for visualizations
import matplotlib.pyplot as plt
import seaborn as sns
# 3. for interaction
from ipywidgets import interact

data = pd.read_csv("data.csv")
# Lets see the top 5 rows of data
data.head()

# Lets find out the number of rows and columns we have in our dataset
data.shape
# Lets find out if there is any null value in our dataset
data.isnull().sum()
# Lets find out the different type of crops in our dataset
data['label'].value_counts()
# Lets find some statistical details of the dataset
data.describe()
# So from this we can see the details of soil and climate for all type of crops
# Let's have a look at the mean
print("Average Ratio of Nitrogen in the soil: {0:.2f}".format(data['N'].mean()))
print("Average Ratio of Phosphorus in the soil: {0:.2f}".format(data['P'].mean()))
print("Average Ratio of Potassium in the soil: {0:.2f}".format(data['K'].mean()))
print("Average Temperature in celcius: {0:.2f}".format(data['temperature'].mean()))
print("Average Relative Humidity in %: {0:.2f}".format(data['humidity'].mean()))
print("Average PH value of the soil: {0:.2f}".format(data['ph'].mean()))
print("Average Rainfall in mm: {0:.2f}".format(data['rainfall'].mean()))

# Now lets check the summary statistics for each of the crop
# This function would print the summary for a given crop
@interact
def summary(crop = list(data['label'].value_counts().index)):
    print(f'Statistics for {crop}:\n')
    conditions = list(data.columns)[:-1]
    renamed_conditions = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'PH', 'Rainfall']
    crop_df = data[data['label'] == crop]
    for renamed_condition, condition in zip(renamed_conditions, conditions):
        print(f'Statistics of {renamed_condition}:')
        print(f'Minimum {renamed_condition} required : ', crop_df[condition].min())
        print(f'Average {renamed_condition} required : ', crop_df[condition].mean())
        print(f'Maximum {renamed_condition} required : ', crop_df[condition].max())
        print()
    print()

# The different types of crop that we have are: 
crops = list(data['label'].value_counts().index)
print(crops)

# Lets check the summary for mango
summary('mango')

# Now lets check the details of soil and climatic conditions required by rice
summary('rice')

# Lets compare the Average Requirements for each crop with average conditions
# This function would print the average condition and the average condition w.r.t each crop

@interact
def average_condition(condition = list(data.columns)[:-1]):
    print(f'Average value for {condition} is {data[condition].mean()}')
    print('----------------------------------------------------------')
    crops = list(data['label'].value_counts().index)
    for crop in crops:
        print(f"For {crop} : {data[data['label']==crop][condition].mean()}")

# The different conditions that we have are: 
conditions = list(data.columns)[:-1]
conditions

# Lets check the average rainfall requirements of each crop
average_condition('rainfall')

# Lets find out crops that require condtions less than the average
# This function would print crops those have condition requirement
# less than the overall average and greater than the overall average.

def compare_with_average(condition):
    print(f"Crops which require greater than average {condition} : ")
    print(f"{data[data[condition] > data[condition].mean()]['label'].unique()}")
    print('-------------------------------------------------------------------')
    print(f"Crops which require leass than average {condition} : ")
    print(f"{data[data[condition] < data[condition].mean()]['label'].unique()}")

conditions = list(data.columns)[:-1]
conditions

# Lets check Nitrogen requirements of the crops
compare_with_average('N')

conditions = list(data.columns)[:-1]
print(conditions)

renamed_conditions = ['Ratio of Nitrogen', 'Ratio of Phosphorus', 'Ration of Potassium', 'Temperature', 'Humidity', 'PH Level', 'Rainfall']
colors = ['orange', 'lightblue', 'lightgreen', 'purple', 'darkgrey', 'darkgreen', 'black']

import warnings
warnings.filterwarnings("ignore")

plt.figure(figsize=(10,5), tight_layout=True)
plt.suptitle('Distribution for Agricultural Conditions', fontsize=20)

for i, condition in enumerate(conditions):
    plt.subplot(2, 4, i+1)
    sns.distplot(data[condition], color=colors[i])
    plt.xlabel(renamed_conditions[i], fontsize=12)

# Lets find some more interesting facts

data.describe()

@interact
def max_condition_requirements(condition=conditions):
    if(condition=='N' or condition=='P' or condition=='K'):
        print(f"Crop which require very High {renamed_conditions[conditions.index(condition)]} in the soil:\
        {data[data[condition]==data[condition].max()]['label'].unique()[0]}")
    else:
        print(f"Crop which require very High {renamed_conditions[conditions.index(condition)]}:\
              {data[data[condition]==data[condition].max()]['label'].unique()[0]}")
        print(f"Crops which require very Low {renamed_conditions[conditions.index(condition)]}:\
              {data[data[condition]==data[condition].min()]['label'].unique()[0]}")

# Now lets find out which crop can be grown only in summer season,
# only in winter season and only in rainy season:

print("Summer Crops: ")
print(data[(data['temperature'] > 30) & (data['humidity'] > 50)]['label'].unique())
print("---------------------------------------------------------------------------")
print("Winter Crops: ")
print(data[(data['temperature'] < 20) & (data['humidity'] > 30)]['label'].unique())
print("---------------------------------------------------------------------------")
print("Rainy Crops: ")
print(data[(data['rainfall'] > 200) & (data['humidity'] > 30)]['label'].unique())

# Now lets move on to build our model
# For this purpose we would use K-means clustering algorithm which belongs
# to the unsupervised machine learning algorithms.
# This would form clusters of crops those which requires same type of soil and
# climatic conditions.
# So first of all lets import the KMeans class from the scikit-learn library
# which performs the K-means clustering.

from sklearn.cluster import KMeans


# we do not need the labels for this algorithm to work so lets drop the labels
x = data.drop(['label'], axis=1)

# selecting all the values of the data
x = x.values

# checking the shape of our data
x.shape


# So, now lets determine the optimal number of clusters within our dataset

plt.rcParams['figure.figsize'] = (10, 4)

wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    km.fit(x)
    wcss.append(km.inertia_)
    
# Lets plot the results
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method', fontsize=20)
plt.xlabel('No. of clusters')
plt.ylabel('wcss')
plt.show()

# Lets implement the K Means algorithm to perform clustering analysis
km = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_means = km.fit_predict(x)

# Lets find out the results
a = data['label']
y_means = pd.DataFrame(y_means)
z = pd.concat([y_means, a], axis=1)
z = z.rename(columns={0: 'cluster'})

# Lets check the clusters of each Crop
print("Lets check the Results After applying the K Means Clustering Analysis \n")
print("Crops in first cluster: ", z[z['cluster']==0]['label'].unique())
print("---------------------------------------------------------------")
print("Crops in second cluster: ", z[z['cluster']==1]['label'].unique())
print("---------------------------------------------------------------")
print("Crops in third cluster: ", z[z['cluster']==2]['label'].unique())
print("---------------------------------------------------------------")
print("Crops in fourth cluster: ", z[z['cluster']==3]['label'].unique())

# Lets split the dataset for Predictive Modelling
# Separate the features and label
y = data['label']
x = data.drop('label', axis=1)

print("Shape of x: ", x.shape)
print("Shape of y: ", y.shape)

# Now lets create Training and Testing sets for Validation purpose
# for this we are going to use the train_test_split class of the sklearn.model_selection module
# Lets import the train_test_split
from sklearn.model_selection import train_test_split

# Now lets use it to split our dataset into training data and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

print("Shape of x train: ", x_train.shape)
print("Shape of x test: ", x_test.shape)
print("Shape of y train: ", y_train.shape)
print("Shape of y test: ", y_test.shape)

# Lets create a Predictive Model
# importing LogisticRegression from the sklearn library
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# Lets evaluate the performance of our model
# importing confusion_matric from the sklearn library
from sklearn.metrics import confusion_matrix

# Lets print the confusion matrix
plt.rcParams['figure.figsize'] = (10, 10)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Wistia')
plt.title('Confusion Matrix for Logistic Regression', fontsize=15)
plt.show()

# Lets print the classification report
from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print(cr)

# Now lets do some real time prediction
# For prediction we need to input the soil and climatic condition into our model
# and the output would be the name of the crop which is best to grow
# in the given soil and climatic condition.
# lets look at some of the input that we can give to our model 
data.head()

prediction = model.predict((np.array([[90, 40, 40, 20, 80, 7, 200]])))
print("The suggested crop for the given soil and climatic conditions is: ", prediction)

