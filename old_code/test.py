## Non optimized RFC and RNN-LSTM integrated

import dash
from dash.dependencies import Output, Input
from dash import dcc, html
from datetime import datetime
import json
import plotly.graph_objs as go
from collections import deque
from flask import Flask, request
import joblib
from math import sqrt
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
#from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_sequences = tf.keras.utils.pad_sequences


# Initialize Flask server and Dash app
server = Flask(__name__)
app = dash.Dash(__name__, server=server)

# Load the machine learning models
rf_model_file = './model/random_forest_model_Z-Y-X-Net-Acc.pkl'
rf_model = joblib.load(rf_model_file)

nn_model_file = './model/lstm_model.h5'
nn_model = tf.keras.models.load_model(nn_model_file, compile=False)
nn_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

# Min-Max Scaler
scaler = MinMaxScaler()
scaler.fit([[0, 0, 0, 0], [1, 1, 1, 1]])  # Adjust these values based on the expected feature range

# Constants
MAX_DATA_POINTS = 1000
UPDATE_FREQ_MS = 100
CSV_FILE_RF = './csv/model_predictions_RandomForestClassifier.csv'
CSV_FILE_NN = './csv/model_predictions_NN_Model.csv'

# Data storage
time = deque(maxlen=MAX_DATA_POINTS)
accel_x = deque(maxlen=MAX_DATA_POINTS)
accel_y = deque(maxlen=MAX_DATA_POINTS)
accel_z = deque(maxlen=MAX_DATA_POINTS)

# Ensure the CSV files exist and have headers
for csv_file in [CSV_FILE_RF, CSV_FILE_NN]:
    if not os.path.isfile(csv_file):
        df = pd.DataFrame(columns=['timestamp', 'x', 'y', 'z', 'net_acc'])
        df.to_csv(csv_file, index=False)

# Layout of the Dash app
app.layout = html.Div(
    [
        dcc.Markdown(
            children="""
            # Live Sensor Readings - Accelerometer
            Streamed from Sensor Logger Mobile...
        """
        ),
        dcc.Graph(id="live_graph"),
        dcc.Interval(id="counter", interval=UPDATE_FREQ_MS),
        html.Div(id='model_input_display', style={'margin-top': '20px'})
    ]
)

# Callback to update the graph and model input display
@app.callback(
    [Output("live_graph", "figure"), Output("model_input_display", "children")],
    Input("counter", "n_intervals")
)
def update_graph(_counter):
    data = [
        go.Scatter(x=list(time), y=list(d), name=name)
        for d, name in zip([accel_x, accel_y, accel_z], ["X", "Y", "Z"])
    ]

    graph = {
        "data": data,
        "layout": go.Layout(
            {
                "xaxis": {"type": "date"},
                "yaxis": {"title": "Acceleration ms<sup>-2</sup>"},
            }
        ),
    }
    
    if len(time) > 0:
        graph["layout"]["xaxis"]["range"] = [min(time), max(time)]
        graph["layout"]["yaxis"]["range"] = [
            min(accel_x + accel_y + accel_z),
            max(accel_x + accel_y + accel_z),
        ]

    # Initialize model input display
    model_input_display = "No significant prediction yet...."

    # Display the latest prediction if available
    if len(time) > 0:
        x_last, y_last, z_last = accel_x[-1], accel_y[-1], accel_z[-1]
        net_acceleration = sqrt(x_last**2 + y_last**2 + z_last**2)
        features = np.array([[x_last, y_last, z_last, net_acceleration]])

        # Apply Min-Max scaling
        scaled_features = scaler.transform(features)

        # Reshape for LSTM
        reshaped_features = pad_sequences([scaled_features], maxlen=100, dtype='float32', padding='pre')

        # Random Forest Model Prediction
        if hasattr(rf_model, 'predict'):
            rf_prediction = rf_model.predict(features)[0]
            if rf_prediction == 1:
                model_input_display  += f"\n Random Forest Model: X = {x_last}, Y = {y_last}, Z = {z_last}, Net_Acc = {net_acceleration:.2f} m/s²"

        # Neural Network Model Prediction
        nn_prediction = nn_model.predict(reshaped_features)[0]
        nn_prediction = int(nn_prediction[0] > 0.5)  # Assuming binary classification with threshold 0.5

        if nn_prediction == 1:
            model_input_display += f"\n Neural Network Model: X = {x_last}, Y = {y_last}, Z = {z_last}, Net_Acc = {net_acceleration:.2f} m/s²"

    return graph, model_input_display

# Flask route to receive and process data
@server.route("/data", methods=["POST"])
def data():
    if request.method == "POST":
        print(f'received data: {request.data}')
        data = json.loads(request.data)
        for d in data['payload']:
            if d.get("name", None) == "accelerometer":
                ts = datetime.fromtimestamp(d["time"] / 1000000000)
                if len(time) == 0 or ts > time[-1]:
                    # Collect data
                    x = d["values"]["x"]
                    y = d["values"]["y"]
                    z = d["values"]["z"]
                    
                    # Update time and accelerometer data
                    time.append(ts)
                    accel_x.append(x)
                    accel_y.append(y)
                    accel_z.append(z)
                    
                    # Prepare feature vector
                    net_acceleration = sqrt(x**2 + y**2 + z**2)
                    features = np.array([[x, y, z, net_acceleration]])

                    # Apply Min-Max scaling
                    scaled_features = scaler.transform(features)

                    # Reshape for LSTM
                    reshaped_features = pad_sequences([scaled_features], maxlen=100, dtype='float32', padding='pre')

                    # Make predictions
                    rf_prediction = rf_model.predict(features)[0] if hasattr(rf_model, 'predict') else None
                    nn_prediction = nn_model.predict(reshaped_features)[0]
                    nn_prediction = int(nn_prediction[0] > 0.5)

                    # Save to CSV if prediction is 1
                    if rf_prediction == 1:
                        new_data_rf = pd.DataFrame([{
                            'timestamp': ts.isoformat(),
                            'x': x,
                            'y': y,
                            'z': z,
                            'net_acc': net_acceleration
                        }])
                        new_data_rf.to_csv(CSV_FILE_RF, mode='a', header=False, index=False)

                    if nn_prediction == 1:
                        new_data_nn = pd.DataFrame([{
                            'timestamp': ts.isoformat(),
                            'x': x,
                            'y': y,
                            'z': z,
                            'net_acc': net_acceleration
                        }])
                        new_data_nn.to_csv(CSV_FILE_NN, mode='a', header=False, index=False)

                    return json.dumps({"timestamp": ts.isoformat(), "x": x, "y": y, "z": z, "net_acc": net_acceleration})

    return "success"

# Run the Dash app
if __name__ == "__main__":
    app.run_server(port=8000, host="0.0.0.0")
