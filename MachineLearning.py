# -*- coding: utf-8 -*-

#Data processing
import pandas as pd

#Cross validation
from sklearn.model_selection import train_test_split

#Data visualization
import matplotlib.pyplot as plt 
import seaborn as sns

#Label encoder
from keras.utils.np_utils import to_categorical

#Linear algebra
import numpy as np

#Metrics score
from sklearn.metrics import confusion_matrix, accuracy_score

#Model algorithms
from keras.callbacks import ReduceLROnPlateau
from sklearn.ensemble import RandomForestClassifier
from keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, Lambda, MaxPool2D
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator


#Dataframes of work
df_train = pd.read_csv('C:/Users/genar/OneDrive/Área de Trabalho/Projetos/Digit_Recognizer/train.csv', engine='python')
df_test = pd.read_csv('C:/Users/genar/OneDrive/Área de Trabalho/Projetos/Digit_Recognizer/test.csv', engine='python')

#Features and target
X_train = (df_train.iloc[:,1:]).values.astype('float32')
X_test = df_test.values.astype('float32')
Y_train = df_train["label"].astype('int32')

#Splitting data to evaluate effectiveness and efficiency of the model
random_seed = 42
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)


"""
*****************************
* Machine learning modeling *
*****************************
"""

#Random Forest
rf = RandomForestClassifier() #no parameters
rf.fit(X_train, Y_train)
pred_rf = rf.predict(X_val)
rf.score(X_train, Y_train) #1.0 // overfitting
#Metrics to evaluate the effectiveness of the model
cmatrix_rf = confusion_matrix(Y_val, pred_rf)
acc_rf = accuracy_score(Y_val, pred_rf) #0.9614285714285714
#Importances
rf_feature = rf.feature_importances_
indices = np.argsort(rf_feature)[::-1]
importances_rf = pd.DataFrame({'Features': indices[:], 'Importance': np.round(rf.feature_importances_, 3)})
#Plot
plt.figure(figsize=(7,3))
plt.plot(indices[:], rf_feature[indices[:]],'k.')
plt.yscale("log")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Random Forest Importances")
plt.savefig("RFImportances.png", size=16)
plt.show()


#Neural-network algorithms

#Redefening features and target to reshape it, if needed
X_train = (df_train.iloc[:,1:]).values.astype('float32')
X_test = df_test.values.astype('float32')
Y_train = df_train["label"].astype('int32')

#Reshaping
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
print(f'Reshape values: {X_train.shape}') #new shape values
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
print(f'Reshape values: {X_test.shape}') #new shape values

#Encode labels to one hot vectors
Y_train= to_categorical(Y_train)
num_classes = Y_train.shape[1]
print(f'Number of classes in train target is: {num_classes}')

#Splitting data to evaluate effectiveness and efficiency of the model
random_seed = 42
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)





#Seeting parameters for neural-networks
max_train = df_train.to_numpy().max() #normalization paramater
def normalize(x): #Normalization function for Lambda layer
    return x/max_train

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)  #Optimization of loss value in accuracy

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', #Sets a learning rate annealer
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.001)


#Linear model
model = Sequential([ #Groups a linear stack of layers
                    Lambda(normalize, input_shape=(28,28,1)), #Arithmetic operation layer
                    Flatten(), #Transforms input into 1D array
                    Dense(10, activation='softmax') #Fully connect layers 
                    ]) 


#Compilement of the neural network
model.compile(optimizer=optimizer,
              loss="categorical_crossentropy",
              metrics=["accuracy"]) #effectiveness measurement

#Data augmentation
gen = ImageDataGenerator()
#Fitting and obtaining accuracy scores
hist = model.fit_generator(gen.flow(X_train, Y_train, batch_size=86), epochs=3,
                              validation_data=(X_val, Y_val), verbose = 2, 
                              steps_per_epoch=X_train.shape[0] // 86,
                              callbacks=[learning_rate_reduction])
hist_dict = hist.history
hist_dict.keys()

#Visualization
#Plot parameters
loss_values = hist_dict['loss']
val_loss_values = hist_dict['val_loss']
acc_values = hist_dict['accuracy']
val_acc_values = hist_dict['val_accuracy']
epochs = range(1, len(loss_values) + 1)
#Plot
plt.style.use('ggplot')
fig, axes = plt.subplots(ncols=2, figsize=(8, 4), dpi=100)
sns.lineplot(epochs, loss_values, label='Loss', ax=axes[0])
sns.lineplot(epochs, val_loss_values, label='Validation Loss', ax=axes[0])
axes[0].set_title('')
axes[0].set_xlabel('Epochs', size=12)
axes[0].set_ylabel('Values', size=12)
sns.lineplot(epochs, acc_values, label='Accuracy', ax=axes[1])
sns.lineplot(epochs, val_acc_values, label='Validation Accuracy', ax=axes[1])
axes[1].set_title('')
axes[1].set_xlabel('Epochs', size=12)
axes[1].set_ylabel('Values', size=12)
plt.suptitle('Linear Model Loss and Accuracy', size=16)
plt.tight_layout()
plt.savefig("LMAccEpochs.png")
plt.show()



#Fully Connected Model
model = Sequential([
        Lambda(normalize, input_shape=(28,28,1)),
        Flatten(),
        Dense(512, activation='relu'),  #Denser layering
        Dense(10, activation='softmax')
        ])

model.compile(optimizer=optimizer,
              loss="categorical_crossentropy",
              metrics=["accuracy"])

hist_fc = model.fit_generator(gen.flow(X_train, Y_train, batch_size=86), epochs=3,
                              validation_data=(X_val, Y_val), verbose = 2, 
                              steps_per_epoch=X_train.shape[0] // 86,
                              callbacks=[learning_rate_reduction])
hist_fc_dict = hist_fc.history
hist_fc_dict.keys()

#Visualization
#Plot parameters
loss_values_fc = hist_fc_dict['loss']
val_loss_values_fc = hist_fc_dict['val_loss']
acc_values_fc = hist_fc_dict['accuracy']
val_acc_values_fc = hist_fc_dict['val_accuracy']
epochs = range(1, len(loss_values_fc) + 1)
#Plot
plt.style.use('ggplot')
fig, axes = plt.subplots(ncols=2, figsize=(8, 4), dpi=100)
sns.lineplot(epochs, loss_values_fc, label='Loss', ax=axes[0])
sns.lineplot(epochs, val_loss_values_fc, label='Validation Loss', ax=axes[0])
axes[0].set_title('')
axes[0].set_xlabel('Epochs', size=12)
axes[0].set_ylabel('Values', size=12)
sns.lineplot(epochs, acc_values_fc, label='Accuracy', ax=axes[1])
sns.lineplot(epochs, val_acc_values_fc, label='Validation Accuracy', ax=axes[1])
axes[1].set_title('')
axes[1].set_xlabel('Epochs', size=12)
axes[1].set_ylabel('Values', size=12)
plt.suptitle('Fully Connected Model Loss and Accuracy', size=16)
plt.tight_layout()
plt.savefig("FCAccEpochs.png")
plt.show()



#Convaluted Neural Network Model
model = Sequential([
                    Lambda(normalize, input_shape=(28,28,1)),
                    Conv2D(filters=32, kernel_size=(5,5), padding='Same', activation='relu'), #convalutional layer
                    Conv2D(filters=32, kernel_size=(5,5), padding='Same', activation='relu'),
                    MaxPool2D(pool_size=(2,2)), #downsampling filtering
                    Dropout(0.25),
                    Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation='relu'),
                    Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation='relu'),
                    MaxPool2D(pool_size=(2,2), strides=(2,2)),
                    Dropout(0.25),
                    Flatten(),
                    Dense(256, activation="relu"),
                    Dropout(0.5),
                    Dense(10, activation="softmax")
                    ])

model.compile(optimizer=optimizer,
              loss="categorical_crossentropy",
              metrics=["accuracy"])

hist_cnn = model.fit(X_train, Y_train, batch_size=86, epochs=3, 
          validation_data=(X_val, Y_val), verbose=2)
hist_cnn_dict = hist_cnn.history
hist_cnn_dict.keys()

#Visualization
#Plot parameters
loss_values_cnn = hist_cnn_dict['loss']
val_loss_values_cnn = hist_cnn_dict['val_loss']
acc_values_cnn = hist_cnn_dict['accuracy']
val_acc_values_cnn = hist_cnn_dict['val_accuracy']
epochs = range(1, len(loss_values_cnn) + 1)
#Plot
plt.style.use('ggplot')
fig, axes = plt.subplots(ncols=2, figsize=(8, 4), dpi=100)
sns.lineplot(epochs, loss_values_cnn, label='Loss', ax=axes[0])
sns.lineplot(epochs, val_loss_values_cnn, label='Validation Loss', ax=axes[0])
axes[0].set_title('')
axes[0].set_xlabel('Epochs', size=12)
axes[0].set_ylabel('Values', size=12)
sns.lineplot(epochs, acc_values_cnn, label='Accuracy', ax=axes[1])
sns.lineplot(epochs, val_acc_values_cnn, label='Validation Accuracy', ax=axes[1])
axes[1].set_title('')
axes[1].set_xlabel('Epochs', size=12)
axes[1].set_ylabel('Values', size=12)
plt.suptitle('CNN Model Loss and Accuracy (No Data Augmentation)', size=16)
plt.tight_layout()
plt.savefig("CNNAccNoAug.png")
plt.show()

#Data augmentation
gen = ImageDataGenerator(rotation_range=8,  #Randomly rotate images in the range
                         width_shift_range=0.08, #Randomly shift images horizontally
                         shear_range=0.3, #Shear angle in counter-clockwise direction in degrees
                         height_shift_range=0.08,  #Randomly shift images vertically 
                         zoom_range=0.08) #Randomly zoom image 
gen.fit(X_train)

#Here we use the same model above already defined but with the data augmentation in training
hist_cnn_aug = model.fit_generator(gen.flow(X_train, Y_train, batch_size=86), epochs=3, #epochs=20 to a more accurate result
                              validation_data=(X_val, Y_val), verbose = 2, 
                              steps_per_epoch=X_train.shape[0] // 86,
                              callbacks=[learning_rate_reduction])

hist_cnn_aug_dict = hist_cnn_aug.history
hist_cnn_aug_dict.keys()

#Visualization
#Plot parameters
loss_values_cnn_aug = hist_cnn_aug_dict['loss']
val_loss_values_cnn_aug = hist_cnn_aug_dict['val_loss']
acc_values_cnn_aug = hist_cnn_aug_dict['accuracy']
val_acc_values_cnn_aug = hist_cnn_aug_dict['val_accuracy']
epochs = range(1, len(loss_values_cnn_aug) + 1)
#Plot
plt.style.use('ggplot')
fig, axes = plt.subplots(ncols=2, figsize=(8, 4), dpi=100)
sns.lineplot(epochs, loss_values_cnn_aug, label='Loss', ax=axes[0])
sns.lineplot(epochs, val_loss_values_cnn_aug, label='Validation Loss', ax=axes[0])
axes[0].set_title('')
axes[0].set_xlabel('Epochs', size=12)
axes[0].set_ylabel('Values', size=12)
sns.lineplot(epochs, acc_values_cnn_aug, label='Accuracy', ax=axes[1])
sns.lineplot(epochs, val_acc_values_cnn_aug, label='Validation Accuracy', ax=axes[1])
axes[1].set_title('')
axes[1].set_xlabel('Epochs', size=12)
axes[1].set_ylabel('Values', size=12)
plt.suptitle('CNN Model Loss and Accuracy (Data Augmented)', size=16)
plt.tight_layout()
plt.savefig("CNNAugAcc.png")
plt.show()

#Fine tune hyperparameters with Batch Normalization and Data Augmentation
model = Sequential([
                    Lambda(normalize, input_shape=(28,28,1)),
                    Conv2D(filters=32, kernel_size=(5,5), padding='Same', activation='relu'),
                    BatchNormalization(axis=1),
                    Conv2D(filters=32, kernel_size=(5,5), padding='Same', activation='relu'),
                    BatchNormalization(axis=1),
                    MaxPool2D(pool_size=(2,2)),
                    Dropout(0.25),
                    BatchNormalization(axis=1),
                    Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation='relu'),
                    BatchNormalization(axis=1),
                    Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation='relu'),
                    MaxPool2D(pool_size=(2,2), strides=(2,2)),
                    Dropout(0.25),
                    Flatten(),
                    BatchNormalization(),
                    Dense(256, activation="relu"),
                    Dropout(0.5),
                    Dense(10, activation="softmax")
                    ])
    
model.compile(optimizer=optimizer, 
              loss="categorical_crossentropy",
              metrics=["accuracy"])

hist_cnnf = model.fit_generator(gen.flow(X_train, Y_train, batch_size=86), epochs=3,
                              validation_data=(X_val, Y_val), verbose = 2, 
                              steps_per_epoch=X_train.shape[0] // 86,
                              callbacks=[learning_rate_reduction])

hist_cnnf_dict = hist_cnnf.history
hist_cnnf_dict.keys() #Batch normalization performed worse than without it


#Visualization
#Plot parameters
loss_values_cnnf = hist_cnnf_dict['loss']
val_loss_values_cnnf = hist_cnnf_dict['val_loss']
acc_values_cnnf = hist_cnnf_dict['accuracy']
val_acc_values_cnnf = hist_cnnf_dict['val_accuracy']
epochs = range(1, len(loss_values_cnnf) + 1)
#Plot
plt.style.use('ggplot')
fig, axes = plt.subplots(ncols=2, figsize=(8, 4), dpi=100)
sns.lineplot(epochs, loss_values_cnnf, label='Loss', ax=axes[0])
sns.lineplot(epochs, val_loss_values_cnnf, label='Validation Loss', ax=axes[0])
axes[0].set_title('')
axes[0].set_xlabel('Epochs', size=12)
axes[0].set_ylabel('Values', size=12)
sns.lineplot(epochs, acc_values_cnnf, label='Accuracy', ax=axes[1])
sns.lineplot(epochs, val_acc_values_cnnf, label='Validation Accuracy', ax=axes[1])
axes[1].set_title('')
axes[1].set_xlabel('Epochs', size=12)
axes[1].set_ylabel('Values', size=12)
plt.suptitle('CNN Model Loss and Accuracy with Batch Normalization', size=16)
plt.tight_layout()
plt.savefig("CNNBNAcc.png")
plt.show()


#Confusion matrix
Y_pred = model.predict(X_val) #using CNN data augmented with no batch normalization model
Y_pred_classes = np.argmax(Y_pred, axis = 1) 
Y_true = np.argmax(Y_val,axis = 1) 
cmatrix = confusion_matrix(Y_true, Y_pred_classes) 
#Plot
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(cmatrix, annot=True, linewidths=0.01, cmap='OrRd', linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label", size=14)
plt.ylabel("True Label", size=14)
plt.title("Confusion Matrix for CNN Model with Data Augmentation", size=16)
plt.savefig("CNNConfMatrix.png")
plt.show()

#Predictions
predictions = model.predict_classes(X_test, verbose=0)
df_predictions = pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
df_predictions.to_csv("predictions.csv", index=False, header=True)