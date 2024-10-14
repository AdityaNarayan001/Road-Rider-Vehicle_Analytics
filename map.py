import pandas as pd
import plotly.express as px

df = pd.DataFrame(pd.read_csv('./csv/TEST__alpha_RFC_pred.csv'))
df.head()

fig = px.density_mapbox(df, lat='latitude', lon='longitude', z='net_acc', radius=10,
                        center=dict(lat=0, lon=180), zoom=2,
                        mapbox_style="open-street-map", width=1700, height=1000)

fig.show()