import dash
from dash.dependencies import Output, Input
import dash_core_components as dcc
import dash_html_components as html
import plotly
import random
import plotly.graph_objs as go
from collections import deque
import pandas as pd
import datetime
import plotly.express as px

app = dash.Dash(__name__)
app.layout = html.Div(
    [   
        html.H3('Vanderbilt University Modeling and Analysis of Complex Systems Lab'),
        dcc.Graph(id='live-graph', animate=True),
        dcc.Interval(
            id='graph-update',
            interval=1*15000
        ),
    ]
)

@app.callback(Output('live-graph', 'figure'), [Input('graph-update', 'n_intervals')])# events=[Event('graph-update', 'interval')])
def update_graph_scatter(_):

    df = pd.read_csv('experience.csv', )
    df['date_time'] = pd.to_datetime(df['time'])
    df.set_index(keys='date_time',inplace=True, drop = True)
    time_end = datetime.datetime.now()
    time_start = time_end-datetime.timedelta(days=2)
    df = df.loc[time_start:time_end,:]

    data1 = go.Scatter(x=list(df['time'].to_numpy().flatten()),
            y=list(df['rlstpt'].to_numpy().flatten()),
            name='Rl-control AHU1 Set Point',
            mode= 'lines+markers',
            )
    data2 = go.Scatter(x=list(df['time'].to_numpy().flatten()),
            y=list(df['oat'].to_numpy().flatten()),
            name='Outside Air Temperature',
            mode= 'lines+markers')
    data3 = go.Scatter(x=list(df['time'].to_numpy().flatten()),
            y=list(df['avg_stpt'].to_numpy().flatten()),
            name='Average Building Zone VRF Set Point',
            mode= 'lines+markers')
    data4 = go.Scatter(x=list(df['time'].to_numpy().flatten()),
            y=list(df['hist_stpt'].to_numpy().flatten()),
            name='Rule Based AHU1 Set Point',
            mode= 'lines+markers')
    data5 = go.Scatter(x=list(df['time'].to_numpy().flatten()),
            y=list(df['oah'].to_numpy().flatten()),
            name='Outside Air Humidity',
            mode= 'lines+markers')
    data6 = go.Scatter(x=list(df['time'].to_numpy().flatten()),
            y=list(df['wbt'].to_numpy().flatten()),
            name='Wet Bulb Temperature',
            mode= 'lines+markers')
    
    myfigure = go.Figure(data = [data1, data2, data3, data4, data5, data6], 
                        layout=go.Layout(
                                    yaxis = {'title' : 'Temperature(F) and Relative Humidity',
                                    'color' : 'black', }, autosize = True,
                                    title = {'text' : 'Demo of PPO RL Controller applied to Alumni Hall AHU',
                                             'font' : {'color' : 'black', 'size' : 24, 'family' : "Times New Roman"}},
                                    hovermode='x'
                                    ), 
                         )

    return myfigure

if __name__ == '__main__':
    # from waitress import serve
    # serve(app, host = '129.59.104.221', port = 8810)
    app.run_server(debug=False, host = '129.59.104.221', port = 8810)

# To run as a safe production server, run it as 
# waitress-serve --host 129.59.104.221 --port 8810 live_plot:app.server

# ps -u nauga (except sudo influxd which we get from ps -auxf)
# kill -9 PID
