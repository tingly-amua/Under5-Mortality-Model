import os
import pickle
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
from flask import Flask, jsonify, request
import dash_bootstrap_components as dbc

# ---------------------------
# Local paths for model & features
# ---------------------------
MODEL_PATH = "final_model.pkl"
FEATURES_PATH = "feature_importances.pkl"

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

# ---------------------------
# Load Model and Features
# ---------------------------
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
# Helper to get top features per target
# ---------------------------
def get_top_features(target, top_n=20):
    df_target = feature_importances[feature_importances["Target"].str.lower() == target.lower()]
    if df_target.empty:
        return []
    return df_target.nlargest(top_n, "Importance")["Feature"].tolist()

def make_dropdown(target):
    return dcc.Dropdown(
        id=f"dropdown-{target.lower()}",
        options=[{"label": f, "value": f} for f in get_top_features(target)],
        placeholder=f"Select features for {target}",
        multi=True
    )

# ---------------------------
# Flask server
# ---------------------------
server = Flask(__name__)

@server.route("/")
def index():
    return """
    <div style="text-align:center;font-family:sans-serif;min-height:100vh;display:flex;flex-direction:column;justify-content:center;align-items:center;color:#004080;">
        <h1>👶 Afya Toto</h1>
        <p>Protecting Children’s Health Through Data Insights</p>
        <a href='/dashboard/'>Go to Dashboard</a>
    </div>
    """

# ---------------------------
# Dash app
# ---------------------------
app = dash.Dash(
    __name__,
    server=server,
    url_base_pathname="/dashboard/",
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

app.layout = dbc.Container([
    dbc.Row([dbc.Col(html.H1("Afya-Toto Dashboard", className="text-center"), width=12)]),

    dbc.Row([
        dbc.Col(make_dropdown("Under-5"), width=3),
        dbc.Col(make_dropdown("Infant"), width=3),
        dbc.Col(make_dropdown("Neonatal"), width=3),
    ], justify="center", className="mb-4"),

    dbc.Row([dbc.Col(html.Button("Predict", id="predict-btn", n_clicks=0, className="btn btn-success"), width="auto")],
            justify="center", className="mb-4"),

    dbc.Row([dbc.Col(html.Div(id="prediction-output", className="text-center"), width=12)])
], fluid=True)

# ---------------------------
# Callback for prediction
# ---------------------------
@app.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    State("dropdown-under-5", "value"),
    State("dropdown-infant", "value"),
    State("dropdown-neonatal", "value")
)
def make_prediction(n_clicks, u5_features, inf_features, neo_features):
    if n_clicks < 1:
        return "ℹ️ Select features first."

    import pandas as pd
    sel = {"Under-5": u5_features or [], "Infant": inf_features or [], "Neonatal": neo_features or []}

    predictions = {}
    for target, features in sel.items():
        if not features:
            predictions[target] = "⚠️ No features selected"
            continue

        # Build dummy input with zeros (replace with actual user input in production)
        X_new = pd.DataFrame([{f: 0 for f in features}])
        model = trained_models.get(target)
        if model is None:
            predictions[target] = "⚠️ Model not loaded"
            continue

        try:
            pred = model.predict(X_new)[0]
            prob = model.predict_proba(X_new)[0, 1]
            predictions[target] = f"Prediction: {pred}, Probability: {prob:.3f}"
        except Exception as e:
            predictions[target] = f"⚠️ Error predicting: {e}"

    return html.Ul([html.Li(f"{t}: {v}") for t, v in predictions.items()])

# ---------------------------
# Run server
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    server.run(debug=True, port=port)
