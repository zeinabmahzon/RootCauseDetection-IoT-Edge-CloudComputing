
import tensorflow as tf 
from tensorflow import keras
from keras.layers import LSTM , Dropout , Dense
# from keras.wrappers.scikit_learn import kerasClassifier
from keras.src.utils import np_utils
# import np_utils
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score 
import pandas as pd
import numpy as np
import os

path='F:\\'

data = pd.read_csv(os.path.join(path,'dataset.csv'))
rec_count=len(data)
sequence_lenght=24
def generate_data(X,y,sequence_lenght=24,step=1):
    X_local=[]
    y_local=[]
    for start in range(0,len(data) - sequence_lenght,step):
        end=start+sequence_lenght
        X_local.append(X[start:end])
        y_local.append(y[end-1])
    return np.array(X_local),np.array(y_local)
X_sequence, y = generate_data(data.loc[:, "sensor_0":"sensor_n"].values, data.source_node)

encoder=LabelEncoder()
encoder.fit(y)
encodedy=encoder.transform(y)
y=np_utils.to_categorical(encodedy)
# print('y',y)

model = keras.Sequential()
model.add(LSTM(100, input_shape = ( 24,25)))
model.add(Dense(26, activation="softmax"))
model.compile(loss="categorical_crossentropy"
                  , metrics =['accuracy']
                  , optimizer="adam")
print("model")
training_size = int(len(X_sequence) * 0.7)
X_train, y_train = X_sequence[:training_size], y[:training_size]
X_test, y_test = X_sequence[training_size:], y[training_size:]
history=model.fit(X_train, y_train, batch_size=24, epochs=40,validation_data=(X_test,y_test))
history_df=pd.DataFrame(history.history)

history_df['epoch'] = history.epoch
history_df['epoch'] = history_df['epoch'] + 1

#loss. 
csv_filename = 'LSTM_training_history.csv'
history_df.to_csv(csv_filename, index=False)
print(f"\nTraining history saved to {csv_filename}")

# accuracy
print("Target ")
print(model.evaluate(X_test, y_test , batch_size=24))



y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# ---F1-Score ---
y_test_lables=np.argmax(y_test,axis=1)

f1_macro = f1_score(y_test_lables, y_pred, average='macro')
print(f"F1-Score (Macro Average): {f1_macro:.4f}")

