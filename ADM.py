import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class AnomalyDetectionModel:
    def __init__(self, history_length=24):
        self.history_length = history_length
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.rmse_threshold = 0.0

    def _prepare_data(self, data):
        X, y = [], []
        for i in range(len(data) - self.history_length):
            X.append(data[i:(i + self.history_length)])
            y.append(data[i + self.history_length])
        return np.array(X), np.array(y)

    def train(self, normal_data: np.ndarray, epochs=40, batch_size=24):
        print("Starting ADM training process...")
        
        scaled_normal_data = self.scaler.fit_transform(normal_data.reshape(-1, 1))
        
        X_train, y_train = self._prepare_data(scaled_normal_data)
        
        self.model = Sequential()
        self.model.add(LSTM(4, input_shape=(self.history_length, 1)))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        
        print(f"Training model on {len(X_train)} normal samples...")
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        
        predictions = self.model.predict(X_train)
        rmse_on_normal = np.sqrt(mean_squared_error(y_train, predictions))
        
        self.rmse_threshold = rmse_on_normal * 2.0
        print(f"Training complete. RMSE threshold for anomaly detection set to: {self.rmse_threshold:.4f}")

    def predict_anomalies(self, test_data: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been trained yet. Please call the train() method first.")
            
        print("\nStarting anomaly detection on test data...")
        
        scaled_test_data = self.scaler.transform(test_data.reshape(-1, 1))
        
        X_test, y_test = self._prepare_data(scaled_test_data)
        
        predictions = self.model.predict(X_test)
        
        anomaly_signals = []
        num_intervals = len(test_data) // 24

        for i in range(num_intervals):
            start_index = i * 24
            interval_predictions = predictions[start_index : start_index + 24]
            interval_actual = y_test[start_index : start_index + 24]
            
            if len(interval_predictions) == 0:
                continue

            interval_rmse = np.sqrt(mean_squared_error(interval_actual, interval_predictions))
            
            if interval_rmse > self.rmse_threshold:
                anomaly_signals.append(1)
                print(f"Interval {i+1}: RMSE = {interval_rmse:.4f} -> Anomaly Detected (1)")
            else:
                anomaly_signals.append(0)
                print(f"Interval {i+1}: RMSE = {interval_rmse:.4f} -> Normal (0)")
                
        return np.array(anomaly_signals)


if __name__ == "__main__":
    normal_len 
    normal_data #INPUT

    test_len 
    test_data #INPUT
 

    adm = AnomalyDetectionModel(history_length=24)
    
    adm.train(normal_data, epochs=50)
    
    binary_signals = adm.predict_anomalies(test_data)
    
    print("\n--- Final Results ---")
    print("Generated binary signals for each 24-hour interval:")
    print(binary_signals)

    plt.figure(figsize=(15, 6))
    plt.plot(normal_data, label="Normal Training Data", color='blue')
    x_axis_test = np.arange(len(normal_data), len(normal_data) + len(test_data))
    plt.plot(x_axis_test, test_data, label="Test Data", color='green')
    
    unique_label_added = False
    for i, signal in enumerate(binary_signals):
        if signal == 1:
            start_idx = len(normal_data) + i * 24
            if not unique_label_added:
                plt.axvspan(start_idx, start_idx + 24, color='red', alpha=0.3, label='Detected Anomaly Interval')
                unique_label_added = True
            else:
                plt.axvspan(start_idx, start_idx + 24, color='red', alpha=0.3)

    plt.title("Anomaly Detection using ADM")
    plt.xlabel("Time (hours)")
    plt.ylabel("Sensor Value")
    plt.legend()
    plt.grid(True)
    plt.show()