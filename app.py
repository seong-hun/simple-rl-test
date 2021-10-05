import numpy as np
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_csv("data/new_out.csv")

app.layout = html.Div([
    dcc.Markdown(
        """
        # PPO for a Point-Mass System with [Dash](https://plotly.com/dash/)
        """
    ),
    html.Hr(),
    html.Div([
        dcc.Graph(
            id="reward-scatter",
            hoverData={'points': [{'customdata': 0}]}
        )
    ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),
    html.Div([
        dcc.Graph(id='pos-time-series'),
        dcc.Graph(id='vel-time-series'),
    ], style={'display': 'inline-block', 'width': '49%'}),

    html.Div(dcc.Slider(
        id="ckpt-slider",
        min=df["ckpt"].min(),
        max=df["ckpt"].max(),
        value=df["ckpt"].max(),
        marks={int(ckpt): str(int(ckpt)) for ckpt in df["ckpt"].unique()},
        step=None,
        updatemode="drag",
    ), style={'width': '49%', 'padding': '0px 20px 20px 20px'})
])


@app.callback(
    dash.dependencies.Output("reward-scatter", "figure"),
    [dash.dependencies.Input("ckpt-slider", "value")])
def update_graph(ckpt_value):
    df["r"] = np.sqrt(df["x"]**2 + df["y"]**2)
    df["theta"] = np.rad2deg(np.arctan2(df["x"], df["y"]))
    df["vr"] = np.sqrt(df["vx"]**2 + df["vy"]**2)
    df["vtheta"] = np.rad2deg(np.arctan2(df["vx"], df["vy"]))

    dff = df[df["ckpt"] == ckpt_value]
    dff0 = dff[dff["t"] == 0]

    d = dff0["return"]
    dfsize = (10 * (d - d.min()) / (d.max() - d.min()) + 30)
    fig = px.scatter_polar(
        dff0, r="r", theta="theta",
        color="return",
        range_color=(1900, 2020),
        size=dfsize,
        color_continuous_scale=px.colors.sequential.Magma,
    )

    fig.update_traces(customdata=dff0["file_index"])
    fig.update_layout(
        margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
        hovermode='closest'
    )

    return fig


@app.callback(
    dash.dependencies.Output('pos-time-series', 'figure'),
    [dash.dependencies.Input('reward-scatter', 'hoverData')],
)
def update_pos_timeseries(hoverData):
    file_index = hoverData['points'][0]['customdata']
    dff = df[df["file_index"] == file_index]
    return create_time_series(dff, ["x", "y"])


@app.callback(
    dash.dependencies.Output('vel-time-series', 'figure'),
    [dash.dependencies.Input('reward-scatter', 'hoverData')],
)
def update_vel_timeseries(hoverData):
    file_index = hoverData['points'][0]['customdata']
    dff = df[df["file_index"] == file_index]
    return create_time_series(dff, ["ux", "uy"])


def create_time_series(dff, columns):

    fig = px.scatter(dff, x="t", y=columns)

    fig.update_traces(mode='lines')
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True, range=(-10, 10))
    fig.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text="Position" if "x" in columns else "Velocity")
    fig.update_layout(
        height=225,
        margin={'l': 20, 'b': 30, 'r': 10, 't': 10},
        plot_bgcolor="white",
    )

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
