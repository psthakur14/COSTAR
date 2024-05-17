# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import tensorflow as tf

from tensorflow import keras


from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from sklearn.model_selection import StratifiedKFold


from sklearn import metrics
from sklearn.metrics import classification_report

# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

import sklearn

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score,roc_auc_score,cohen_kappa_score, matthews_corrcoef

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, auc, precision_recall_curve
from imblearn.metrics import geometric_mean_score
from imblearn.keras import BalancedBatchGenerator
from sklearn.utils.class_weight import compute_sample_weight
from keras import backend as K

from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC

from sklearn.datasets import make_classification



import torch
import torch.nn as nn
import torch.optim as optim

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import SelectFromModel

from tensorflow.keras.callbacks import LearningRateScheduler

import csv
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

df= pd.read_csv('CSV_CodeBert_Output_combined.csv')


df.drop('Unnamed: 0',axis=1, inplace= True)

df



def extract_number(text):
    match = re.search(r'_(\d+)_', text)
    if match:
        return int(match.group(1))
    return None

# df['severity'] = df['dir_name'].apply(extract_number)
df['severity'] = df.apply(lambda x: 4 if x['severity'] in ['critical'] else 3 if x['severity'] in ['major'] else 2 if x['severity'] in ['minor'] else 1, axis = 1)
df = df.rename(columns={'smell':'dir_name'})

print("DF shape ",df.shape)



print(df['Embedding'].loc[0])

df = df[df['0'].notna()]

df['severity'] = df.apply(lambda x: 1 if x['severity'] in [3,4,2] else 0, axis = 1)
df = df.rename(columns={'severity':'label'})

# df.dropna(subset=['Embedding'],inplace=True)

df_godClass= df[df['dir_name'].str.contains('blob')]
df_dataClass= df[df['dir_name'].str.contains('data class')]
df_featureEnvy= df[df['dir_name'].str.contains('feature envy')]
df_longMethod= df[df['dir_name'].str.contains('long method')]

print("df_godClass ",df_godClass['label'].value_counts())
print("df_dataClass ",df_dataClass['label'].value_counts())
print("df_featureEnvy ",df_featureEnvy['label'].value_counts())
print("df_longMethod ",df_longMethod['label'].value_counts())





print("overall shape ",df.shape," \nlabels counts ",df['label'].value_counts())
print("df_godClass ", df_godClass.shape," \nlabels counts ",df_godClass['label'].value_counts())
print("df_dataClass ", df_dataClass.shape," \nlabels counts ",df_dataClass['label'].value_counts())
print("df_featureEnvy ", df_featureEnvy.shape," \nlabels counts ",df_featureEnvy['label'].value_counts())
print("df_longMethod ", df_longMethod.shape," \nlabels counts ",df_longMethod['label'].value_counts())

df_dataClass.reset_index(inplace=True,drop=True)
df_godClass.reset_index(inplace=True,drop=True)
df_featureEnvy.reset_index(inplace=True,drop=True)
df_longMethod.reset_index(inplace=True,drop=True)

df_list=[df_dataClass,df_godClass,df_featureEnvy,df_longMethod]
col=df.columns

for df_l in df_list:
  i=0
  for c in col:
    if df_l[c].dtype=='O':
      print(c)
      if c!='dataset':
        i=i+1
        df_l.drop(c,axis=1,inplace=True)
  print("Total Colunm deleted: ",i)

print("overall shape ",df.shape," \nlabels counts ",df['label'].value_counts())
print("df_godClass ", df_godClass.shape," \nlabels counts ",df_godClass['label'].value_counts())
print("df_dataClass ", df_dataClass.shape," \nlabels counts ",df_dataClass['label'].value_counts())
print("df_featureEnvy ", df_featureEnvy.shape," \nlabels counts ",df_featureEnvy['label'].value_counts())
print("df_longMethod ", df_longMethod.shape," \nlabels counts ",df_longMethod['label'].value_counts())

print(df_dataClass)

"""#ML techniques"""


models = {}

#Logistic Regression

models['Logistic Regression'] = LogisticRegression(max_iter=10000)

#Linear SVM
models['Support Vector Machines'] = LinearSVC(max_iter=30000)

#Decision Tree
models['Decision Trees'] = DecisionTreeClassifier()

#Random Forest
models['Random Forest'] = RandomForestClassifier()

#Naiv bayse
models['Naive Bayes'] = GaussianNB()

#K-NearestNeighbour
models['K-Nearest Neighbor'] = KNeighborsClassifier()

# Create and train AdaBoost classifier
models['Ada Boost'] = AdaBoostClassifier(n_estimators=100, random_state=42)

# Create and train Gradient Boosting classifier
models['Gradiant Boosting'] = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Create and train XGBoost classifier
models['XG Boost'] = xgb.XGBClassifier(n_estimators=100, random_state=42)

"""#others"""

# Create an MLP classifier
models['Multilayer Perceptron'] = MLPClassifier(hidden_layer_sizes=(100,))

# Create a QDA classifier
models['Quadratic Discriminant Analysis'] = QuadraticDiscriminantAnalysis()



def my_model(X_train, y_train,model):
  model.fit(X_train, y_train)
  return model


def prediction(X_test,y_test, model,key, dataset ):
  columns=['Dataset','classifier','accuracy','precision','recall','f1-score','Cohen_Kappa','MCC','AUCPR','TP','FP','FN','TN']
  # print("Prediction start")
  #perform prediction
  total=len(X_test)

  p_hat = model.predict(X_test)
  y_hat = np.where(p_hat > 0.5, 1, 0)
  #print(y_hat)
  #tn, fp, fn, tp = confusion_matrix(y_test,y_hat).ravel()
  result1= metrics.classification_report(y_test,y_hat, output_dict=True,labels=[0,1],digits=4)
  kappa_value= sklearn.metrics.cohen_kappa_score(y_test,y_hat)
  mcc_value=sklearn.metrics.matthews_corrcoef(y_test,y_hat)
  cm=metrics.confusion_matrix(y_test,y_hat, labels=[0,1])
  accuracy = accuracy_score(y_test,y_hat)
  precision = precision_score(y_test,y_hat, average='macro')
  recall = recall_score(y_test,y_hat,average='macro')
  f1 = f1_score(y_test,y_hat,average='macro')
  # y_pred_proba = model.predict_proba(X_test)[:, 1]
  # aucprc = roc_auc_score(y_test, y_pred_proba)
  aucprc_value = average_precision_score(y_test,y_hat,average='macro',)
  
  TP = cm[0][0]
  FP = cm[0][1]
  FN = cm[1][0]
  TN = cm[1][1]
  # print("prediction Done")
  df_temp = pd.DataFrame(columns=columns)
  df_temp.loc[0] = [dataset, key, accuracy, precision, recall, f1, kappa_value, mcc_value, aucprc_value, TP, FP, FN, TN]
  return df_temp

"""#end"""



def plot_pca(data, labels):
    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)

    # Plot PCA results with color-coded labels
    plt.figure(figsize=(5, 4))
    plt.scatter(pca_result[labels == 0, 0], pca_result[labels == 0, 1], color='red', label='Label 0', alpha=0.5)
    plt.scatter(pca_result[labels == 1, 0], pca_result[labels == 1, 1], color='blue', label='Label 1', alpha=0.5)

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Plot')
    plt.legend()
    plt.grid(True)
    plt.show()
def plot_pca_y(data, labels):
    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)

    # Plot PCA results with color-coded labels
    plt.figure(figsize=(5, 4))

    # View from the y-axis perspective
    plt.scatter(pca_result[labels == 0, 1], pca_result[labels == 0, 0], color='red', label='Label 0', alpha=0.5)
    plt.scatter(pca_result[labels == 1, 1], pca_result[labels == 1, 0], color='blue', label='Label 1', alpha=0.5)

    # Adjust the labels accordingly
    plt.xlabel('Principal Component 2')
    plt.ylabel('Principal Component 1')

    plt.title('PCA Plot (View from Y-Axis)')
    plt.legend()
    plt.grid(True)
    plt.show()


# Example usage
df_list_pca=[df_dataClass,df_godClass,df_featureEnvy,df_longMethod]
for df_pca in df_list_pca:
  if df_pca is df_dataClass:
    print('Data Class')
  if df_pca is df_godClass:
    print('God Class')
  if df_pca is df_featureEnvy:
    print('Feature Envy')
  if df_pca is df_longMethod:
    print('Long Method')

  X_pca=df_pca.copy()
  y_pca=df_pca['label'].copy()
  X_pca.drop('label',axis=1,inplace=True)
  # data = np.random.randn(100, 5)  # Replace with your data
  # labels = np.random.choice([0, 1], size=100)  # Replace with your labels

  plot_pca(X_pca, y_pca)
  plot_pca_y(X_pca, y_pca)

  



"""#Combining SG, Ada Boost and gradiant Boosting"""

# Combine predictions using voting or other ensemble techniques
def combine_xg_ada_gb(X_train, y_train, X_test, y_test, dataset, key):
  columns_ens=['Dataset','classifier','accuracy','precision','recall','f1-score','Cohen_Kappa','MCC','AUCPR','TP','FP','FN','TN']
  # For example, you can use a majority vote:
  ada_clf = AdaBoostClassifier(n_estimators=100, random_state=42)
  ada_clf.fit(X_train, y_train)

  # Create and train Gradient Boosting classifier
  gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
  gb_clf.fit(X_train, y_train)

  # Create and train XGBoost classifier
  xgb_clf = xgb.XGBClassifier(n_estimators=100, random_state=42)
  xgb_clf.fit(X_train, y_train)

  # Predict using each classifier
  ada_predictions = ada_clf.predict(X_test)
  gb_predictions = gb_clf.predict(X_test)
  xgb_predictions = xgb_clf.predict(X_test)

  ensemble_predictions = (ada_predictions + gb_predictions + xgb_predictions) >= 2

  # # Evaluate ensemble classifier's performance
  # Evaluate the ensemble model
  kappa_value= sklearn.metrics.cohen_kappa_score(y_test,ensemble_predictions)
  mcc_value=sklearn.metrics.matthews_corrcoef(y_test,ensemble_predictions)
  cm=metrics.confusion_matrix(y_test,ensemble_predictions, labels=[0,1])
  accuracy = accuracy_score(y_test,ensemble_predictions)
  precision = precision_score(y_test,ensemble_predictions, average='macro')
  recall = recall_score(y_test,ensemble_predictions,average='macro')
  f1 = f1_score(y_test,ensemble_predictions,average='macro')
  # y_pred_proba = model.predict_proba(X_test)[:, 1]
  # aucprc = roc_auc_score(y_test, y_pred_proba)
  aucprc_value = average_precision_score(y_test,ensemble_predictions,average='macro',)

  TP = cm[0][0]
  FP = cm[0][1]
  FN = cm[1][0]
  TN = cm[1][1]
  # print("prediction Done")
  df_temp = pd.DataFrame(columns=columns_ens)
  df_temp.loc[0] = [dataset, key, accuracy, precision, recall, f1, kappa_value, mcc_value, aucprc_value, TP, FP, FN, TN]

  return df_temp

"""#Ensemble classifier"""

# Load and prepare data (assuming X_train, y_train, X_test, y_test are already defined)
def ensemble_classifiers(X_train, y_train, X_test, y_test, dataset, key):
  columns_en=['Dataset','classifier','accuracy','precision','recall','f1-score','Cohen_Kappa','MCC','AUCPR','TP','FP','FN','TN']

  # Initialize individual models
  lr_model = LogisticRegression()
  svm_model = SVC(probability=True)  # Set probability=True to enable soft voting
  rf_model = RandomForestClassifier()

  # Create a voting classifier
  voting_classifier = VotingClassifier(
      estimators=[('lr', lr_model), ('svm', svm_model), ('rf', rf_model)],
      voting='soft'  # Use 'soft' voting to take into account class probabilities
  )

  # Train the voting classifier on the training data
  voting_classifier.fit(X_train, y_train)

  # Make predictions using the voting classifier
  ensemble_pred = voting_classifier.predict(X_test)

  # Calculate the accuracy of the ensemble model
  accuracy = accuracy_score(y_test, ensemble_pred)
  print(f"Ensemble Accuracy: {accuracy:.4f}")

  # Evaluate the ensemble model
  kappa_value= sklearn.metrics.cohen_kappa_score(y_test,ensemble_pred)
  mcc_value=sklearn.metrics.matthews_corrcoef(y_test,ensemble_pred)
  cm=metrics.confusion_matrix(y_test,ensemble_pred, labels=[0,1])
  accuracy = accuracy_score(y_test,ensemble_pred)
  precision = precision_score(y_test,ensemble_pred, average='macro')
  recall = recall_score(y_test,ensemble_pred,average='macro')
  f1 = f1_score(y_test,ensemble_pred,average='macro')
  # y_pred_proba = model.predict_proba(X_test)[:, 1]
  # aucprc = roc_auc_score(y_test, y_pred_proba)
  aucprc_value = average_precision_score(y_test,ensemble_pred,average='macro',)

  TP = cm[0][0]
  FP = cm[0][1]
  FN = cm[1][0]
  TN = cm[1][1]
  # print("prediction Done")
  df_temp = pd.DataFrame(columns=columns_en)
  df_temp.loc[0] = [dataset, key, accuracy, precision, recall, f1, kappa_value, mcc_value, aucprc_value, TP, FP, FN, TN]

  return df_temp

"""#Ensemble neural network or Random Neural Network"""

def random_neural_network(X_train, y_train, X_test, y_test, dataset, key, num_networks = 5):
  columns_rnn=['Dataset','classifier','accuracy','precision','recall','f1-score','Cohen_Kappa','MCC','AUCPR','TP','FP','FN','TN']
  # Initialize a list to store the ensemble of models
  ensemble_models = []

  # Train and create the ensemble of models
  for _ in range(num_networks):
      model = tf.keras.Sequential([
          tf.keras.layers.Input(shape=(X_train.shape[1],)),
          tf.keras.layers.Dense(128, activation='relu'),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(64, activation='relu'),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(32, activation='relu'),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(1, activation='sigmoid')
      ])

      model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

      model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

      ensemble_models.append(model)

  # Make predictions using the ensemble
  ensemble_predictions = np.zeros_like(y_test, dtype=float)  # Initialize an array to store ensemble predictions

  for model in ensemble_models:
      model_predictions = model.predict(X_test)
      ensemble_predictions += model_predictions.squeeze()  # Aggregate predictions

  ensemble_predictions /= num_networks  # Average the predictions

  # Convert the average predictions to binary predictions
  ensemble_binary_predictions = (ensemble_predictions >= 0.5).astype(int)

  # Evaluate the model
  kappa_value= sklearn.metrics.cohen_kappa_score(y_test,ensemble_binary_predictions)
  mcc_value=sklearn.metrics.matthews_corrcoef(y_test,ensemble_binary_predictions)
  cm=metrics.confusion_matrix(y_test,ensemble_binary_predictions, labels=[0,1])
  accuracy = accuracy_score(y_test,ensemble_binary_predictions)
  precision = precision_score(y_test,ensemble_binary_predictions, average='macro')
  recall = recall_score(y_test,ensemble_binary_predictions,average='macro')
  f1 = f1_score(y_test,ensemble_binary_predictions,average='macro')
  # y_pred_proba = model.predict_proba(X_test)[:, 1]
  # aucprc = roc_auc_score(y_test, y_pred_proba)
  aucprc_value = average_precision_score(y_test,ensemble_binary_predictions,average='macro',)

  TP = cm[0][0]
  FP = cm[0][1]
  FN = cm[1][0]
  TN = cm[1][1]
  # print("prediction Done")
  df_temp = pd.DataFrame(columns=columns_rnn)
  df_temp.loc[0] = [dataset, key, accuracy, precision, recall, f1, kappa_value, mcc_value, aucprc_value, TP, FP, FN, TN]

  return df_temp

"""#NN prediction Model"""

def define_model(n_input):

    model = tf.keras.Sequential([
          tf.keras.layers.Input(shape=(n_input,)),
          tf.keras.layers.Dense(512, activation='relu'),
          tf.keras.layers.Dropout(0.1),
          tf.keras.layers.Dense(1, activation='sigmoid')
      ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

#isko phir se karo.. yaha par slack variable ka flow chart bana kar

def train_nn(x_train, y_train, x_test, y_test, num_ep=100):

  num_epochs=num_ep

  input_size = x_train.shape[1]
  hidden_size = [128, 64] #128  # Adjust as needed
  output_size = 1   # Adjust based on your problem (regression/classification)

  model_nns = define_model(input_size)
  IR= max(np.bincount(y_train))/ min(np.bincount(y_train))
  weights = {0:1, 1:IR}
  history = model_nns.fit(x_train, y_train, class_weight=weights, epochs=num_ep, verbose=0)
  # evaluate model
  predicted_output = model_nns.predict(x_test)
  # score = roc_auc_score(model_nns, yhat)
  # print("predicted_output ",predicted_output)
  predicted_labels = (predicted_output >= 0.5)

  return predicted_labels

#Function to calculate evaluation matrics
def predict_nn(y_test_nn, y_hat_nn,key_nn,dataset_nn):
  columns_nn=['Dataset','classifier','accuracy','precision','recall','f1-score','Cohen_Kappa','MCC','AUCPR','TP','FP','FN','TN']
  # result1= metrics.classification_report(y_test_nn,y_hat_nn, output_dict=True,labels=[0,1],digits=4)


  kappa_value_nn= sklearn.metrics.cohen_kappa_score(y_test_nn,y_hat_nn)
  mcc_value_nn=sklearn.metrics.matthews_corrcoef(y_test_nn,y_hat_nn)
  cm_nn=metrics.confusion_matrix(y_test_nn,y_hat_nn, labels=[0,1])
  accuracy_nn = accuracy_score(y_test_nn,y_hat_nn)
  precision_nn = precision_score(y_test_nn,y_hat_nn, average='macro')
  recall_nn = recall_score(y_test_nn,y_hat_nn,average='macro')
  f1_nn = f1_score(y_test_nn,y_hat_nn,average='macro')
  # y_pred_proba = model.predict_proba(X_test)[:, 1]
  # aucprc = roc_auc_score(y_test, y_pred_proba)
  aucprc_value_nn = average_precision_score(y_test_nn,y_hat_nn,average='macro',)

  TP_nn = cm_nn[0][0]
  FP_nn = cm_nn[0][1]
  FN_nn = cm_nn[1][0]
  TN_nn = cm_nn[1][1]

  df_temp_nn = pd.DataFrame(columns=columns_nn)
  df_temp_nn.loc[0] = [dataset_nn, key_nn, accuracy_nn, precision_nn, recall_nn, f1_nn, kappa_value_nn, mcc_value_nn, aucprc_value_nn, TP_nn, FP_nn, FN_nn, TN_nn]

  return df_temp_nn


# Define a simple neural network function
def create_neural_network(input_dim):
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_dim,)),
        # Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model



class KerasEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, class_weight=None, input_dim=768,epochs=150, verbose=0):
        self.input_dim = input_dim
        self.model = create_neural_network(self.input_dim)
        self.class_weight = class_weight
        self.epochs = epochs
        self.verbose = verbose

    def fit(self, X, y):
        if self.class_weight is not None:
            sample_weights = np.array([self.class_weight[class_label] for class_label in y])
        else:
            sample_weights = None

        self.model.fit(X, y, sample_weight=sample_weights)
        return self

    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype(int)

def train_ada_nn(X_train, y_train, X_test, y_test_wnn, class_weights,dataset_wnn, key_wnn):

  # Create a custom Keras classifier wrapper
  input_dim = X_train.shape[1]
  # keras_classifier = KerasClassifierWrapper(build_fn=create_neural_network, input_dim=input_dim, epochs=50, verbose=0)
  IR= max(np.bincount(y_train))/ min(np.bincount(y_train))
  class_weights = {0: 1.0, 1: IR}  # Adjust the weights as needed
  keras_classifier = KerasEstimator(class_weight=class_weights,input_dim=input_dim)

  # Create an AdaBoost classifier with the custom Keras-based estimator as the base estimator

  adaboost_classifier = AdaBoostClassifier(n_estimators=300, learning_rate=0.001, random_state=42, algorithm='SAMME')
  # Train the AdaBoost classifier
  adaboost_classifier.fit(X_train, y_train)

  # Make predictions on the test data
  y_hat_wnn = adaboost_classifier.predict(X_test)

  # predicted_output = adaboost_classifier.predict(X_test)

  # y_hat_wnn = (predicted_output > 0.5)#.astype(int)


  columns_wnn=['Dataset','classifier','accuracy','precision','recall','f1-score','Cohen_Kappa','MCC','AUCPR','TP','FP','FN','TN']

  kappa_value_nn= sklearn.metrics.cohen_kappa_score(y_test_wnn,y_hat_wnn)
  mcc_value_nn=sklearn.metrics.matthews_corrcoef(y_test_wnn,y_hat_wnn)
  cm_nn=metrics.confusion_matrix(y_test_wnn,y_hat_wnn, labels=[0,1])
  accuracy_nn = accuracy_score(y_test_wnn,y_hat_wnn)
  precision_nn = precision_score(y_test_wnn,y_hat_wnn, average='macro')
  recall_nn = recall_score(y_test_wnn,y_hat_wnn,average='macro')
  f1_nn = f1_score(y_test_wnn,y_hat_wnn,average='macro')
  # y_pred_proba = model.predict_proba(X_test)[:, 1]
  # aucprc = roc_auc_score(y_test, y_pred_proba)
  aucprc_value_nn = average_precision_score(y_test_wnn,y_hat_wnn,average='macro',)

  TP_nn = cm_nn[0][0]
  FP_nn = cm_nn[0][1]
  FN_nn = cm_nn[1][0]
  TN_nn = cm_nn[1][1]

  df_temp_wnn = pd.DataFrame(columns=columns_wnn)
  df_temp_wnn.loc[0] = [dataset_wnn, key_wnn, accuracy_nn, precision_nn, recall_nn, f1_nn, kappa_value_nn, mcc_value_nn, aucprc_value_nn, TP_nn, FP_nn, FN_nn, TN_nn]

  return df_temp_wnn





def build_model(input_size):
    # Create a neural network model
    model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_size,)),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    return model

def lr_schedule(epoch):
    initial_learning_rate = 0.001
    if epoch < 150:
        return initial_learning_rate
    elif epoch < 300:
        return initial_learning_rate / 10.0
    else:
        return initial_learning_rate / 100.0

# Define a weighted binary cross-entropy loss function
def weighted_binary_crossentropy(class_weights):
    def loss(y_true, y_pred):
        # Cast y_true to the same data type as y_pred
        y_true = tf.cast(y_true, y_pred.dtype)

        # Calculate weighted binary cross-entropy
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        weighted_losses = - (class_weights[1] * y_true * tf.math.log(y_pred) + class_weights[0] * (1 - y_true) * tf.math.log(1 - y_pred))
        return tf.reduce_mean(weighted_losses)
    return loss




def auc_pr(y_true, y_pred):
    # Ensure y_true and y_pred are NumPy arrays
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()

    # Calculate precision and recall at multiple thresholds
    precision_values, recall_values, _ = precision_recall_curve(y_true, y_pred)

    # Calculate AUC-PR using Scikit-Learn's auc function
    auc_pr_value = auc(recall_values, precision_values)

    return auc_pr_value

def train_wnn(X_train, y_train, X_test, y_test_wnn, class_weights, dataset_wnn, key_wnn):
  input_size = X_train.shape[1]
  model= build_model(input_size)


  model.compile(optimizer=Adam(learning_rate=0.001), loss=weighted_binary_crossentropy(class_weights), metrics=[tf.keras.metrics.AUC(curve="PR", name="auc_pr", multi_label=False)])

  model.fit(X_train, y_train, epochs=750, batch_size=32, validation_data=(X_test, y_test_wnn))
  # Evaluate the model
  predicted_output = model.predict(X_test)
  # print("y_test ",y_test_wnn)
  print("predicted_output ",predicted_output)
  th1= min(np.bincount(y_train))/ max(np.bincount(y_train))
  th=0.5-th1
  y_hat_wnn = (predicted_output > th)#.astype(int)
  print("threshold ",th," y_hat_wnn ",y_hat_wnn)

  columns_wnn=['Dataset','classifier','accuracy','precision','recall','f1-score','Cohen_Kappa','MCC','AUCPR','TP','FP','FN','TN']

  kappa_value_nn= sklearn.metrics.cohen_kappa_score(y_test_wnn,y_hat_wnn)
  mcc_value_nn=sklearn.metrics.matthews_corrcoef(y_test_wnn,y_hat_wnn)
  cm_nn=metrics.confusion_matrix(y_test_wnn,y_hat_wnn, labels=[0,1])
  accuracy_nn = accuracy_score(y_test_wnn,y_hat_wnn)
  precision_nn = precision_score(y_test_wnn,y_hat_wnn, average='macro')
  recall_nn = recall_score(y_test_wnn,y_hat_wnn,average='macro')
  f1_nn = f1_score(y_test_wnn,y_hat_wnn,average='macro')
  # y_pred_proba = model.predict_proba(X_test)[:, 1]
  # aucprc = roc_auc_score(y_test, y_pred_proba)
  aucprc_value_nn = average_precision_score(y_test_wnn,y_hat_wnn,average='macro',)

  TP_nn = cm_nn[0][0]
  FP_nn = cm_nn[0][1]
  FN_nn = cm_nn[1][0]
  TN_nn = cm_nn[1][1]

  df_temp_wnn = pd.DataFrame(columns=columns_wnn)
  df_temp_wnn.loc[0] = [dataset_wnn, key_wnn, accuracy_nn, precision_nn, recall_nn, f1_nn, kappa_value_nn, mcc_value_nn, aucprc_value_nn, TP_nn, FP_nn, FN_nn, TN_nn]

  return df_temp_wnn



"""#Training and Testing"""

csv_file='Mean_Results.csv'
csv_all_fold= 'AllFold_results.csv'


Headding_list= ['Dataset','classifier','accuracy','precision','recall','f1-score','Cohen_Kappa','MCC','AUCPR','TP','FP','FN','TN','Positive', 'Negative']
# open the file in the write mode

f = open(csv_file, 'w')

# create the csv writer
writer = csv.writer(f)

# write a row to the csv file
writer.writerow(Headding_list)

columns=['Dataset','classifier','accuracy','precision','recall','f1-score','Cohen_Kappa','MCC','AUCPR','TP','FP','FN','TN','Positive', 'Negative']
imb_df = pd.DataFrame(columns=columns)
imb_df = imb_df.fillna(0)
result_imb = pd.DataFrame(columns=columns)
result_imb = result_imb.fillna(0)




RANDOM_STATE = None

dataset='Hi'

df_lists= [df_dataClass,df_godClass,df_featureEnvy,df_longMethod]
for df_ls in df_lists:

  if df_ls is df_dataClass:
    dataset= 'Data Class'
  if df_ls is df_godClass:
    dataset= 'God Class'
  if df_ls is df_featureEnvy:
    dataset= 'Feature Envy'
  if df_ls is df_longMethod:
    dataset= 'Long Method'

  # if df_ls is df_dataClass:
  #   continue
  # if df_ls is df_godClass:
  #   continue
  # if df_ls is df_longMethod:
  #   break
  print("-----------------------------Dataset - ",dataset," ------------------------------------")
  n_folds=10
  fold=0
  df_temp= df_ls.copy()


  for i in tqdm(range(n_folds)):
      train, test = train_test_split(df_temp,stratify=df_temp['label'], test_size=0.2,random_state=1)
      print("\ntrain ", train.shape)
      print("test ", test.shape)
      train.reset_index(drop=True, inplace=True)
      test.reset_index(drop=True, inplace=True)
      print("After index reseting train ", train.shape)
      print("After index reseting test ", test.shape)

      print("Instance counts in test dataset  ", test['label'].value_counts())
      x_train= train.copy()
      y_train= train['label']
      x_train.drop('label',axis=1,inplace=True)

      x_test= test.copy()
      y_test= test['label']
      x_test.drop('label',axis=1,inplace=True)

      print("----------- Fold ",fold," of dataset ",dataset," --------------")
      fold+=1

      res_imb_df = pd.DataFrame(columns=columns)
      res_imb_df = res_imb_df.fillna(0)
      for key in models.keys():

                print("|| results of ",key, end=" ")
                model1=my_model(x_train.copy(), y_train.copy(),models[key])
                # Prediction
                size_of_data1=len(x_test)
                # TP,FP,FN,TN,org_accuracy,org_precision,org_recall,org_f1,org_kappa_value,org_mcc_value,org_aucprc=prediction(x_test.copy,y_test.copy, 'Original',size_of_data1,model1)
                df2imb=prediction(x_test,y_test, model1,key, dataset )

                df2imb['Positive']=min(np.bincount(y_train))
                df2imb['Negative']=max(np.bincount(y_train))

                res_imb_df=pd.concat([res_imb_df,df2imb])


      # print("Performing Concatenation of the all classifiers resulst")
      imb_df=pd.concat([imb_df,res_imb_df])

      # -----------combining GG, Ada, GB-----------------------------------
      print("|| Execution of the combine_xg_ada_gb", end=" ")
      df_combine= combine_xg_ada_gb(x_train, y_train, x_test, y_test, dataset, 'Combined Ada Xg GB')
      df_combine['Positive']=min(np.bincount(y_train))
      df_combine['Negative']=max(np.bincount(y_train))

      imb_df=pd.concat([imb_df,df_combine])

      # ------------Ensemble Classifiers----------------------------------
      print("|| Execution of the Ensemble Classifier", end=" ")
      df_ensemble= ensemble_classifiers(x_train, y_train, x_test, y_test, dataset, 'Ensemble Classifier')
      df_ensemble['Positive']=min(np.bincount(y_train))
      df_ensemble['Negative']=max(np.bincount(y_train))

      imb_df=pd.concat([imb_df,df_ensemble])

      #-----------OurNeural Network---------------------
      print("|| Neural Network", end=" ")
      x_tr=x_train.copy()
      y_tr=y_train.copy()
      x_tst= x_test.copy()
      y_tst= y_test.copy()

      predicted_labels= train_nn(x_tr, y_tr, x_tst, y_tst, num_ep=300)

      df_nn= predict_nn(y_tst, predicted_labels,'Neural Network',dataset)
      df_nn['Positive']=min(np.bincount(y_tr))
      df_nn['Negative']=max(np.bincount(y_tr))

      imb_df=pd.concat([imb_df,df_nn])

      # ----------Random Neural Networks----------------------------------
      print("|| Execution of the Random Neural Networks", end=" ")
      df_rnn= random_neural_network(x_train, y_train, x_test, y_test, dataset, 'Random Neural Network', num_networks = 10)
      df_rnn['Positive']=min(np.bincount(y_train))
      df_rnn['Negative']=max(np.bincount(y_train))

      imb_df=pd.concat([imb_df,df_rnn])

      # ----------weighted Neural Network---------------------------------
      print("\n|| Execution of the Weighted Neural Network", end=" ")
      class_weights = compute_sample_weight(class_weight='balanced', y=y_tr)
      df_wnn= train_wnn(x_tr, y_tr, x_tst, y_tst, class_weights, dataset, 'Weighted Neural Network')

      df_wnn['Positive']=min(np.bincount(y_tr))
      df_wnn['Negative']=max(np.bincount(y_tr))

      imb_df=pd.concat([imb_df,df_wnn])

      # ----------Adaboost Neural Network---------------------------------
      print("\n|| Execution of the Adaboost Neural Network", end=" ")
      class_weights = compute_sample_weight(class_weight='balanced', y=y_tr)
      df_adann= train_ada_nn(x_tr, y_tr, x_tst, y_tst, class_weights,dataset, 'Adaboost Neural Network')

      df_adann['Positive']=min(np.bincount(y_tr))
      df_adann['Negative']=max(np.bincount(y_tr))

      imb_df=pd.concat([imb_df,df_adann])
  # break

print(imb_df)

# result_imb[(result_imb['classifier'] =='Weighted Neural Network')& (result_imb['Dataset']=='Data Class')]

ds_list=['Data Class','God Class','Feature Envy','Long Method']
for da in ds_list:
  avg= [imb_df[(imb_df['classifier']=='Logistic Regression') & (imb_df['Dataset']==da)].mean()]
  avg_df = pd.DataFrame(avg)
  avg_df['classifier']='Logistic Regression'
  avg_df['Dataset']=da
  result_imb=result_imb.append(avg_df)
  # print(res_imb_df)
  avg= [imb_df[(imb_df['classifier']=='Support Vector Machines') & (imb_df['Dataset']==da)].mean()]
  avg_df = pd.DataFrame(avg)
  avg_df['classifier']='Support Vector Machines'
  avg_df['Dataset']=da
  result_imb=result_imb.append(avg_df)

  avg= [imb_df[(imb_df['classifier']=='Decision Trees') & (imb_df['Dataset']==da)].mean()]
  avg_df = pd.DataFrame(avg)
  avg_df['classifier']='Decision Trees'
  avg_df['Dataset']=da
  result_imb=result_imb.append(avg_df)

  avg= [imb_df[(imb_df['classifier']=='Random Forest') & (imb_df['Dataset']==da)].mean()]
  avg_df = pd.DataFrame(avg)
  avg_df['classifier']='Random Forest'
  avg_df['Dataset']=da
  result_imb=result_imb.append(avg_df)

  avg= [imb_df[(imb_df['classifier']=='Naive Bayes') & (imb_df['Dataset']==da)].mean()]
  avg_df = pd.DataFrame(avg)
  avg_df['classifier']='Naive Bayes'
  avg_df['Dataset']=da
  result_imb=result_imb.append(avg_df)

  avg= [imb_df[(imb_df['classifier']=='K-Nearest Neighbor') & (imb_df['Dataset']==da)].mean()]
  avg_df = pd.DataFrame(avg)
  avg_df['classifier']='K-Nearest Neighbor'
  avg_df['Dataset']=da
  result_imb=result_imb.append(avg_df)


  avg= [imb_df[(imb_df['classifier']=='Ada Boost') & (imb_df['Dataset']==da)].mean()]
  avg_df = pd.DataFrame(avg)
  avg_df['classifier']='Ada Boost'
  avg_df['Dataset']=da
  result_imb=result_imb.append(avg_df)

  avg= [imb_df[(imb_df['classifier']=='Gradiant Boosting') & (imb_df['Dataset']==da)].mean()]
  avg_df = pd.DataFrame(avg)
  avg_df['classifier']='Gradiant Boosting'
  avg_df['Dataset']=da
  result_imb=result_imb.append(avg_df)

  avg= [imb_df[(imb_df['classifier']=='XG Boost') & (imb_df['Dataset']==da)].mean()]
  avg_df = pd.DataFrame(avg)
  avg_df['classifier']='XG Boost'
  avg_df['Dataset']=da
  result_imb=result_imb.append(avg_df)

  avg= [imb_df[(imb_df['classifier']=='Multilayer Perceptron') & (imb_df['Dataset']==da)].mean()]
  avg_df = pd.DataFrame(avg)
  avg_df['classifier']='Multilayer Perceptron'
  avg_df['Dataset']=da
  result_imb=result_imb.append(avg_df)

  avg= [imb_df[(imb_df['classifier']=='Quadratic Discriminant Analysis') & (imb_df['Dataset']==da)].mean()]
  avg_df = pd.DataFrame(avg)
  avg_df['classifier']='Quadratic Discriminant Analysis'
  avg_df['Dataset']=da
  result_imb=result_imb.append(avg_df)

  avg= [imb_df[(imb_df['classifier']=='Neural Network') & (imb_df['Dataset']==da)].mean()]
  avg_df = pd.DataFrame(avg)
  avg_df['classifier']='Neural Network'
  avg_df['Dataset']=da
  result_imb=result_imb.append(avg_df)

  avg= [imb_df[(imb_df['classifier']=='Ensemble Classifier') & (imb_df['Dataset']==da)].mean()]
  avg_df = pd.DataFrame(avg)
  avg_df['classifier']='Ensemble Classifier'
  avg_df['Dataset']=da
  result_imb=result_imb.append(avg_df)

  avg= [imb_df[(imb_df['classifier']=='Combined Ada Xg GB') & (imb_df['Dataset']==da)].mean()]
  avg_df = pd.DataFrame(avg)
  avg_df['classifier']='Combined Ada Xg GB'
  avg_df['Dataset']=da
  result_imb=result_imb.append(avg_df)

  avg= [imb_df[(imb_df['classifier']=='Random Neural Network') & (imb_df['Dataset']==da)].mean()]
  avg_df = pd.DataFrame(avg)
  avg_df['classifier']='Random Neural Network'
  avg_df['Dataset']=da
  result_imb=result_imb.append(avg_df)


  # avg= [imb_df[(imb_df['classifier']=='CNN') & (imb_df['Dataset']==da)].mean()]
  # avg_df = pd.DataFrame(avg)
  # avg_df['classifier']='CNN'
  # avg_df['Dataset']=da
  # result_imb=result_imb.append(avg_df)

  avg= [imb_df[(imb_df['classifier']=='Weighted Neural Network') & (imb_df['Dataset']==da)].mean()]
  avg_df = pd.DataFrame(avg)
  avg_df['classifier']='Weighted Neural Network'
  avg_df['Dataset']=da
  result_imb=result_imb.append(avg_df)

print(result_imb)



result_imb.to_csv(csv_file, mode='a', index=False, header=False)
imb_df.to_csv(csv_all_fold, mode='a', index=False, header=False)

print("Done!!!")








