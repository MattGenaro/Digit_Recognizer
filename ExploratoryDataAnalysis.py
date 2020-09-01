# -*- coding: utf-8 -*-

#Data processing
import pandas as pd

#Data visualization
import matplotlib.pyplot as plt 
import seaborn as sns

#Linear algebra
import numpy as np

#Dataframes of work
df_train = pd.read_csv('/Digit_Recognizer/train.csv'')
df_test = pd.read_csv('/Digit_Recognizer/test.csv')

#First look into 'train' data
df_train.head(5)
df_train.shape
df_train.index
df_train.columns
df_train.describe()

#Checking data 'clearness'
df_train.isnull().sum() #no missing values
df_train.isnull().any().describe()
max_df_train = df_train.max() #detecting outliers and mislabeled values
min_df_train = df_train.min()
df_train.to_numpy().max()
df_train[df_train > 255]

#First look into 'test' data
df_test.head(5)
df_test.shape
df_test.index
df_test.columns
df_test.describe()

#Checking data 'clearness'
df_test.isnull().sum() #no missing values
df_test.isnull().any().describe()
max_df_test = df_test.max() #detecting outliers and mislabeled values
min_df_test = df_test.min()
max_test = df_test.to_numpy().max()
df_test[df_test > 255]


#Visualization of data
df_train_array = np.array(df_train.iloc[:,1:])
df_test_array = np.array(df_test)
df_test_array = df_test_array.reshape(df_test.shape[0], 28, 28, 1)
#Plot
plt.figure(figsize=(5,5))
for i in range(0,64):
    plt.subplot(8, 8, i+1)
    grid_data = df_train_array[i].reshape(28,28)  #reshape from 1D to 2D pixel array
    plt.imshow(grid_data, interpolation = "none", cmap = "bone_r")
    plt.xticks([])
    plt.yticks([])
plt.suptitle('28x28 data samples', size=18)
plt.savefig('DataSamples.png')
plt.show()

#Features and target
X_train = (df_train.iloc[:,1:]).values.astype('float32')
X_test = df_test.values.astype('float32')
Y_train = df_train["label"].astype('int32')
Y_train.value_counts() #amount of each value in 'label' attribute

#Visualization of 'label' counts
plt.style.use('ggplot')
sns.countplot(Y_train, alpha=0.6)
plt.title('Amount of each value in Label')
plt.savefig('LabelCount.png')
plt.show()



