# TODO : Make data window to pass to model and add other models too.
# Done : Integrate Lat,Log data.

import dash
from dash.dependencies import Output, Input
from dash import dcc, html, dcc
from datetime import datetime
import json
import plotly.graph_objs as go
from collections import deque
from flask import Flask, request
import joblib
import os
import pandas as pd
from math import sqrt
import numpy as np

server = Flask(__name__)
app = dash.Dash(__name__, server=server)

model_file = "./model/random_forest_model_Z-Y-X-Net-Acc.pkl"
model = joblib.load(model_file)

MAX_DATA_POINTS = 1000
UPDATE_FREQ_MS = 100
CSV_FILE = "./csv/TEST__alpha_RFC_pred.csv"

time = deque(maxlen=MAX_DATA_POINTS)
accel_x = deque(maxlen=MAX_DATA_POINTS)
accel_y = deque(maxlen=MAX_DATA_POINTS)
accel_z = deque(maxlen=MAX_DATA_POINTS)
pred_arr_RFC = deque(maxlen=1)

lat_long = {
    'latitude':0,
    'longitude':0
    }

if not os.path.isfile(CSV_FILE):
    df = pd.DataFrame(columns=['timestap','acc_x','acc_y','acc_z','net_acc','latitude','longitude'])
    df.to_csv(CSV_FILE, index=False)

app.layout = html.Div(
    [
        dcc.Markdown(
            children="""
            # Live Sensor Readings - Accelerometer
            ## Streamed from Sensor Logger Mobile...
        """
        ),
        html.Hr(),
        dcc.Graph(id="live_graph"),
        dcc.Interval(id="counter", interval=UPDATE_FREQ_MS),
        html.Hr(),
        html.Div(id='model_input_display', style={'margin-top': '20px'})
    ]
)

@app.callback(
        Output("live_graph", "figure"), Output("model_input_display", "children"),
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
    if (
        len(time) > 0
    ):  #  cannot adjust plot ranges until there is at least one data point
        graph["layout"]["xaxis"]["range"] = [min(time), max(time)]
        graph["layout"]["yaxis"]["range"] = [
            min(accel_x + accel_y + accel_z),
            max(accel_x + accel_y + accel_z),
        ]

    model_input_display = "No Detection ..."
    if len(pred_arr_RFC) == 1:
        model_input_display = f"Detection for Timestamp : {pred_arr_RFC[0][0]}, X : {pred_arr_RFC[0][1]}, Y : {pred_arr_RFC[0][2]}, Z : {pred_arr_RFC[0][3]}, Net. Acc : {pred_arr_RFC[0][4]} m/s²"

    return graph, model_input_display

@server.route("/data", methods=["POST"])
def data():
    if str(request.method) == "POST":
        # print(f'received data: {request.data}')
        data = json.loads(request.data)
        for d in data['payload']:
            if d.get("name", None) == "location":
                lat_long['latitude'] = d["values"]["latitude"]
                lat_long['longitude'] = d["values"]["longitude"]
            if (
                d.get("name", None) == "accelerometer"
            ):  #  modify to access different sensors
                ts = datetime.fromtimestamp(d["time"] / 1000000000)
                if len(time) == 0 or ts > time[-1]:
                    x = (d["values"]["x"])
                    y = (d["values"]["y"])
                    z = (d["values"]["z"])
                    
                    time.append(ts)
                    accel_x.append(x)
                    accel_y.append(y)
                    accel_z.append(z)
                    
                    net_acceleration = sqrt(x**2 + y**2 + z**2)
                    features = np.array([[z, y, x, net_acceleration]])
                    
                    if hasattr(model, "predict"):
                        prediction = model.predict(features)[0]
                    else:
                        raise ValueError("Model does not have method PREDICT")
                    
                    if prediction == 1:
                        new_data = pd.DataFrame([{
                            "timestamp":ts.isoformat(),
                            "x":x,
                            "y":y,
                            "z":z,
                            "net_acc":net_acceleration,
                            "latitude":lat_long["latitude"],
                            "longitude":lat_long["longitude"]
                        }])
                        new_data.to_csv(CSV_FILE, mode='a', header=False, index=False)
                        pred_arr_RFC.append([ts.isoformat(), x, y, z, net_acceleration])
                        

    return "success"


if __name__ == "__main__":
    app.run_server(port=8000, host="0.0.0.0")