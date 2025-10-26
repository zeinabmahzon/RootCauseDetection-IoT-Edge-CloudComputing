
# First XGBoost model for Pima Indians dataset
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , f1_score
# load data
dataset = loadtxt('F:\\data.csv', delimiter=",")
# split data into X and y
X = dataset[:,0:n]
Y = dataset[:,n]
# split data into train and test sets
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
# fit model on training data
model = XGBClassifier()
model.fit(X_train, y_train)

# make predictions for test data
y_pred = model.predict(X_test)

predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


f1 = f1_score(y_test, y_pred, average='macro')
print(f"F1-Score (Macro Average): {f1:.4f}")

