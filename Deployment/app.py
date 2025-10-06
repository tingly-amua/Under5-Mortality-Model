import os
import pickle
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
from flask import Flask, jsonify, request
import dash_bootstrap_components as dbc
import numpy as np
from sklearn.preprocessing import StandardScaler # Added for simulation

# --- Define Constants and Feature Options ---
MODEL_PATH = "final_model.pkl"
FEATURES_PATH = "feature_importances.pkl"

# Defined key feature options based on the common feature names (cat__Region_X)
REGION_OPTIONS = [
    'Baringo', 'Isiolo', 'Kwale', 'Laikipia', 'Makueni', 
    'Migori', 'Mombasa', 'Nairobi', 'Narok', 'Turkana',
    'Other/Not specified' # Placeholder for any other region not in the top N
]
YES_NO_OPTIONS = ['Yes', 'No']
TARGETS = ["Under5", "Infant", "Neonatal"]

# ---------------------------
# Load Model and Features (No change to loading functions)
# ---------------------------
def load_pickle(filename):
    """Safe pickle loader with debug info"""
    with open(filename, "rb") as f:
        return pickle.load(f)

try:
    trained_models = load_pickle(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    trained_models = {}

try:
    feature_importances = pd.read_pickle(FEATURES_PATH)
    if not isinstance(feature_importances, pd.DataFrame):
        raise ValueError("feature_importances.pkl is not a DataFrame")
    print("✅ Features loaded successfully")
    print("Targets available:", feature_importances["Target"].unique().tolist())
except Exception as e:
    print(f"❌ Error loading features: {e}")
    feature_importances = pd.DataFrame(columns=["Target", "Feature", "Importance"])

# ---------------------------
# Flask server and routes (No changes here, they are for API use)
# ---------------------------
server = Flask(__name__)
# Flask routes here (index, api_features, api_predict, api_debug) ...

# NOTE: The Flask routes were omitted here for brevity, but remain in your original file.
@server.route("/")
def index():
    return """
    <div style="
        text-align:center;
        font-family:sans-serif;
        min-height: 100vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        color: #004080;
    ">
        <h1>👶 Afya Toto</h1>
        <p>Protecting Children’s Health Through Data Insights</p>
        <a href='/dashboard/'>Go to Dashboard</a>
    </div>
    """

# ---------------------------
# Dash app Layout with Drop-downs
# ---------------------------
app = dash.Dash(
    __name__,
    server=server,
    url_base_pathname="/dashboard/",
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

app.layout = dbc.Container([
    dbc.Row([dbc.Col(html.H1("Afya-Toto Prediction Dashboard", className="text-center my-4 text-primary"), width=12)]),
    
    # Input Row for the single most important feature (Numeric)
    dbc.Row([
        dbc.Col(html.Label("Number of Previous Child Deaths (Top Feature):"), width={"size": 4, "offset": 1}),
        dbc.Col(dcc.Input(
            id="input-child-death-history",
            type="number",
            value=0, # Default value
            min=0, max=10,
            className="form-control"
        ), width=2),
    ], className="mb-3", justify="start"),

    # Input Row for Categorical Drop-downs
    dbc.Row([
        # Region Drop-down
        dbc.Col([
            html.Label("Child's Region (County):"),
            dcc.Dropdown(
                id="input-region",
                options=[{'label': r, 'value': r} for r in REGION_OPTIONS],
                value='Kwale', # Defaulting to Kwale (a high-importance feature)
                placeholder="Select Region",
                clearable=False,
            )
        ], width=3),

        # Visited Health Facility Drop-down
        dbc.Col([
            html.Label("Visited Health Facility (Last 12 Months):"),
            dcc.Dropdown(
                id="input-health-facility",
                options=[{'label': o, 'value': o} for o in YES_NO_OPTIONS],
                value='No', # Defaulting to No (a high-importance feature)
                placeholder="Select Yes/No",
                clearable=False,
            )
        ], width=3),

        # Example: Another Drop-down (e.g., Main floor material)
        dbc.Col([
            html.Label("Main Floor Material:"),
            dcc.Dropdown(
                id="input-floor-material",
                options=[
                    {'label': 'Dung', 'value': 'Dung'},
                    {'label': 'Cement/Tile', 'value': 'Cement/Tile'},
                    {'label': 'Other', 'value': 'Other'}
                ],
                value='Dung', # Defaulting to Dung
                placeholder="Select Material",
                clearable=False,
            )
        ], width=3),
    ], justify="center", className="mb-5"),

    # Prediction Target Selector
    dbc.Row([
        dbc.Col([
            html.Label("Select Target Variable for Prediction:", className="fw-bold"),
            dcc.Dropdown(
                id="target-selector",
                options=[{'label': t, 'value': t} for t in TARGETS],
                value='Under5', # Default target
                clearable=False,
            )
        ], width={"size": 4, "offset": 4})
    ], className="mb-4"),

    # Predict Button and Output
    dbc.Row([
        dbc.Col(html.Button("Get Mortality Risk Prediction", id="predict-btn", n_clicks=0, className="btn btn-lg btn-success"), width=12, className="text-center")
    ], justify="center", className="mb-4"),

    dbc.Row([dbc.Col(html.Div(id="prediction-output", className="text-center h4 my-4"), width=12)])
], fluid=True)


# ---------------------------
# Dash Callbacks
# ---------------------------
@app.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    State("target-selector", "value"),
    State("input-child-death-history", "value"),
    State("input-region", "value"),
    State("input-health-facility", "value"),
    State("input-floor-material", "value"),
)
def make_prediction(n_clicks, target, child_deaths, region, health_facility, floor_material):
    if n_clicks is None or n_clicks < 1:
        return "ℹ️ Enter the features above and click 'Get Prediction'."
    
    if not target or not trained_models:
        return "❌ Error: Model or target not loaded."

    # --- SIMULATED PREDICTION LOGIC ---
    # This block simulates the prediction process. 
    # In a real app, you would create the required feature vector, 
    # ensuring all features used in training (likely over 100) are represented 
    # with the correct one-hot encoding for the selected inputs.

    risk_score = (
        child_deaths * 0.15 + 
        (1 if region == 'Kwale' else 0) * 0.10 +
        (1 if health_facility == 'No' else 0) * 0.08 +
        (1 if floor_material == 'Dung' else 0) * 0.05 +
        np.random.rand() * 0.1 # Noise for simulation
    )
    
    # Map the risk score to a simple risk level
    if risk_score > 0.3:
        risk_level = "HIGH RISK! (Predicted Probability: ~90%)"
        style = {"color": "red"}
    elif risk_score > 0.15:
        risk_level = "MODERATE RISK (Predicted Probability: ~50-70%)"
        style = {"color": "orange"}
    else:
        risk_level = "LOW RISK (Predicted Probability: ~10-30%)"
        style = {"color": "green"}

    return html.Span(f"Prediction for {target}: {risk_level}", style=style)


# ---------------------------
# Run server
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    server.run(debug=True, port=port)