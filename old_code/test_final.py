## Non optimised RFC integrated

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

# Initialize Flask server and Dash app
server = Flask(__name__)
app = dash.Dash(__name__, server=server)

# Load the machine learning model using joblib
model_file = './model/random_forest_model_Z-Y-X-Net-Acc.pkl'
model = joblib.load(model_file)

# Check model type and methods
print(f"Model type: {type(model)}")
print(f"Model methods: {dir(model)}")

# Constants
MAX_DATA_POINTS = 1000
UPDATE_FREQ_MS = 100
CSV_FILE = './csv/model_predictions_RandomForestClassifier.csv'

# Data storage
time = deque(maxlen=MAX_DATA_POINTS)
accel_x = deque(maxlen=MAX_DATA_POINTS)
accel_y = deque(maxlen=MAX_DATA_POINTS)
accel_z = deque(maxlen=MAX_DATA_POINTS)

# Ensure the CSV file exists and has headers
if not os.path.isfile(CSV_FILE):
    df = pd.DataFrame(columns=['timestamp', 'x', 'y', 'z', 'net_acc'])
    df.to_csv(CSV_FILE, index=False)

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
    model_input_display = "No significant prediction yet."

    # Display the latest prediction if available
    if len(time) > 0 and hasattr(model, 'predict'):
        # Prepare feature vector for prediction
        net_acceleration = sqrt(accel_x[-1]**2 + accel_y[-1]**2 + accel_z[-1]**2)
        features = np.array([[accel_z[-1], accel_y[-1], accel_x[-1], net_acceleration]])
        prediction = model.predict(features)[0]
        
        if prediction == 1:
            # Update model input display
            model_input_display = f"Model input values: X={accel_x[-1]}, Y={accel_y[-1]}, Z={accel_z[-1]}, Net_Acc={net_acceleration:.2f} m/s²"

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
                    features = np.array([[z, y, x, net_acceleration]])
                    
                    # Make prediction
                    if hasattr(model, 'predict'):
                        prediction = model.predict(features)[0]
                    else:
                        raise ValueError("Loaded model does not have a predict method")
                    
                    # Save to CSV if prediction is 1
                    if prediction == 1:
                        new_data = pd.DataFrame([{
                            'timestamp': ts.isoformat(),
                            'x': x,
                            'y': y,
                            'z': z,
                            'net_acc': net_acceleration
                        }])
                        new_data.to_csv(CSV_FILE, mode='a', header=False, index=False)
                        return json.dumps({"timestamp": ts.isoformat(), "x": x, "y": y, "z": z, "net_acc": net_acceleration})
                    
    return "success"

# Run the Dash app
if __name__ == "__main__":
    app.run_server(port=8000, host="0.0.0.0")
