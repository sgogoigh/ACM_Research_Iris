# Prepare a logistic regression model and an Artificial Neural Network
# to classify features from the Iris dataset. 
# Explain in detail the various data-preprocessing techniques used, 
# performance metrics for both the models and plot graphs for the same.

#Using logistic regression

#readying to import iris dataset
import csv
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#for plotting graphs
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import seaborn as sns
sns.set(style = "white")

#storing the dataset
iris = load_iris()
#print(iris)

#iris.keys()
# OP - dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])

#iris.target_names
# OP - array(['setosa', 'versicolor', 'virginica'], dtype='<U10')

#iris.feature_names
# OP - ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

# There are three types of iris flowers - setosa, versicolor and virginica
# The purpose of the logistic regression model is identify which set of data
# depicts which flower

# Organising the data frame

iris_df = pd.DataFrame(iris.data)
iris_df.columns = iris.feature_names
#iris_df.head()

# We now prepare the input and output data
# X - Input
# Y - Output
X = iris.data
Y = iris.target

#print(X.shape)
# Input data has 150 rows and 4 columns
#print(Y.shape)
# Target data has 150 rows

# Splitting of training data and test data is required
# For training machine learning algorithm, model.fit() is used
# After training, we predict based on the new input given
# After prediction, we compare using model.score() for accuracy

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X,Y,test_size = 0.2, random_state = 2)

#starting with logistic regression
lorg = LogisticRegression(random_state = 0)
lorg.fit(X_train, Y_train)

# For predictions
Y_pred = lorg.predict(X_test)

# Now we use a CONFUSION MATRIX which checks the correctness or accuracy
# of our predictions

check = confusion_matrix(Y_test, Y_pred)

# TO check the accuracy, we use function accuracy_score
acc = accuracy_score(Y_test,Y_pred)
print(acc)