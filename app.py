import os
import pickle
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from flask import Flask, request, jsonify

# ---------------------------
# Load trained models (all targets)
# ---------------------------
with open("all_models.pkl", "rb") as f:
    models = pickle.load(f)

# Example: models = {"Under-5": model1, "Infant": model2, "Neonatal": model3}

# ---------------------------
# Flask server
# ---------------------------
server = Flask(__name__)

@server.route("/")
def index():
    return "Welcome to the Mortality Prediction API and Dashboard!"

@server.route("/api/predict/<target>", methods=["POST"])
def predict_api(target):
    """
    Flask API endpoint to get predictions.
    Expects JSON payload with 'features': [..list of feature values..]
    """
    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "Invalid request. 'features' key is required."}), 400

    if target not in models:
        return jsonify({"error": f"Target '{target}' not found. Use one of {list(models.keys())}"}), 400

    model = models[target]
    features = [data["features"]]
    prediction = model.predict(features).tolist()

    return jsonify({"target": target, "prediction": prediction})

# ---------------------------
# Dash app for visualization
# ---------------------------
app = dash.Dash(__name__, server=server, url_base_pathname="/dashboard/")

app.layout = html.Div(
    style={
        "fontFamily": "Arial, sans-serif",
        "maxWidth": "800px",
        "margin": "auto",
        "padding": "20px",
        "backgroundColor": "#f9f9f9",
        "borderRadius": "8px",
        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
    },
    children=[
        html.H1("Mortality Prediction Dashboard", style={"textAlign": "center", "color": "#333"}),

        html.P("Enter feature values (comma separated) and select target outcome:", 
               style={"textAlign": "center", "color": "#666"}),

        dcc.Input(
            id="features-input",
            placeholder="Enter features, e.g. 0.5,1.2,3.4",
            style={"width": "100%", "padding": "10px", "fontSize": "16px"},
        ),

        dcc.Dropdown(
            id="target-dropdown",
            options=[{"label": k, "value": k} for k in models.keys()],
            placeholder="Select a target (Under-5, Infant, Neonatal)",
            style={"marginTop": "10px"},
        ),

        html.Button(
            "Predict",
            id="predict-button",
            n_clicks=0,
            style={
                "display": "block",
                "margin": "20px auto",
                "padding": "10px 20px",
                "fontSize": "16px",
                "cursor": "pointer",
                "backgroundColor": "#007BFF",
                "color": "white",
                "border": "none",
                "borderRadius": "5px",
            },
        ),

        dcc.Loading(
            id="loading-spinner",
            type="circle",
            children=dcc.Graph(id="prediction-chart"),
        ),
    ],
)

# ---------------------------
# Callback to update chart
# ---------------------------
@app.callback(
    Output("prediction-chart", "figure"),
    [Input("predict-button", "n_clicks")],
    [State("features-input", "value"), State("target-dropdown", "value")],
)
def update_chart(n_clicks, features_value, target_value):
    if n_clicks == 0 or not features_value or not target_value:
        return go.Figure().update_layout(
            title_text='Enter features + select target, then click "Predict"',
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[{
                "text": "Waiting for input...",
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 16, "color": "#888"},
            }],
        )

    try:
        features = [float(x.strip()) for x in features_value.split(",")]
        model = models[target_value]
        pred = model.predict([features]).tolist()[0]
    except Exception as e:
        return go.Figure().update_layout(
            title_text="Error",
            annotations=[{
                "text": str(e),
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 14, "color": "red"},
            }],
        )

    fig = go.Figure(data=[go.Bar(
        x=[target_value],
        y=[pred],
        marker_color="steelblue",
        text=[str(pred)],
        textposition="auto"
    )])

    fig.update_layout(
        title_text=f"Prediction for {target_value}",
        xaxis_title="Target",
        yaxis_title="Predicted Value",
        plot_bgcolor="white",
        paper_bgcolor="#f9f9f9",
        font={"color": "#333"},
    )
    return fig

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    server.run(debug=False, port=port, host="0.0.0.0")
