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

# ---------------------------
# Load pickled objects
# ---------------------------
def load_pickle(filename):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    else:
        print(f"❌ File not found: {filename}")
        return None

# Load final model
trained_models = load_pickle(MODEL_PATH)
model_loaded = trained_models is not None
if model_loaded:
    print("✅ Model loaded successfully")

# Load feature importances
feature_importances = load_pickle(FEATURES_PATH)
features_loaded = False
available_targets = []
sample_rows = []
feature_importances_shape = [0, 0]
feature_importances_columns = ["Target", "Feature", "Importance"]

if isinstance(feature_importances, pd.DataFrame):
    feature_importances_shape = list(feature_importances.shape)
    available_targets = feature_importances["Target"].unique().tolist() if "Target" in feature_importances.columns else []
    sample_rows = feature_importances.head().to_dict(orient="records")
    features_loaded = True
    print("✅ Features loaded successfully")
    print("Available targets:", available_targets)
else:
    print("❌ feature_importances.pkl is not a DataFrame")
    feature_importances = pd.DataFrame(columns=feature_importances_columns)

# ---------------------------
# Extract top features dynamically
# ---------------------------
def get_top_features(target, top_n=20):
    if feature_importances.empty or target not in feature_importances["Target"].values:
        return []
    filtered = feature_importances[feature_importances["Target"].str.lower() == target.lower()]
    if filtered.empty:
        return []
    return filtered.nlargest(top_n, "Importance")["Feature"].tolist()

def make_dropdown(target):
    return dcc.Dropdown(
        id=f"dropdown-{target.lower()}",
        options=[{"label": f, "value": f} for f in get_top_features(target)],
        placeholder=f"Select features for {target}",
        multi=True
    )

# ---------------------------
# Flask server and routes
# ---------------------------
server = Flask(__name__)

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

@server.route("/api/features", methods=["GET"])
def api_features():
    if feature_importances.empty:
        return jsonify({"error": "❌ feature_importances is empty"})
    result = {}
    for t in feature_importances["Target"].unique():
        feats = (
            feature_importances[feature_importances["Target"] == t]
            .nlargest(20, "Importance")["Feature"]
            .tolist()
        )
        result[t] = feats
    return jsonify(result)

@server.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.json
    # fallback: simple response if trained_models not callable
    if callable(trained_models.get("Under5", None)):
        prediction = trained_models["Under5"](data)
    else:
        prediction = "⚠️ Model not loaded or invalid"
    return jsonify({"prediction": prediction})

@server.route("/api/debug", methods=["GET"])
def api_debug():
    """Return debug info for models and features"""
    debug_info = {
        "model_loaded": model_loaded,
        "features_loaded": features_loaded,
        "feature_importances_shape": feature_importances_shape,
        "feature_importances_columns": feature_importances_columns,
        "available_targets": available_targets,
        "sample_rows": sample_rows
    }
    return jsonify(debug_info)

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

# Update to match exact target names from DataFrame
TARGETS = available_targets or ["Under5", "Infant", "Neonatal"]

app.layout = dbc.Container([
    dbc.Row([dbc.Col(html.H1("Afya-Toto Dashboard", className="text-center"), width=12)]),
    
    dbc.Row([dbc.Col(make_dropdown(t), width=3) for t in TARGETS], justify="center", className="mb-4"),
    
    dbc.Row([dbc.Col(html.Button("Predict", id="predict-btn", n_clicks=0, className="btn btn-success"), width="auto")],
            justify="center", className="mb-4"),
    
    dbc.Row([dbc.Col(html.Div(id="prediction-output", className="text-center"), width=12)])
], fluid=True)

# ---------------------------
# Callback to populate dropdowns dynamically
# ---------------------------
@app.callback(
    Output("dropdown-under5", "options"),
    Output("dropdown-infant", "options"),
    Output("dropdown-neonatal", "options"),
    Input("predict-btn", "n_clicks")
)
def populate_dropdowns(_):
    under5_opts = [{"label": f, "value": f} for f in get_top_features("Under5")]
    infant_opts = [{"label": f, "value": f} for f in get_top_features("Infant")]
    neo_opts = [{"label": f, "value": f} for f in get_top_features("Neonatal")]
    return under5_opts, infant_opts, neo_opts

# ---------------------------
# Callback for prediction
# ---------------------------
@app.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    State("dropdown-under5", "value"),
    State("dropdown-infant", "value"),
    State("dropdown-neonatal", "value")
)
def make_prediction(n_clicks, u5_features, inf_features, neo_features):
    if n_clicks < 1:
        return "ℹ️ Select features first."

    sel = {"Under5": u5_features or [], "Infant": inf_features or [], "Neonatal": neo_features or []}
    predictions = {}

    for target, features in sel.items():
        if not features:
            predictions[target] = "⚠️ No features selected"
            continue

        X_new = pd.DataFrame([{f: 0 for f in features}])  # Replace with actual user input in production
        model = trained_models.get(target) if trained_models else None
        if model is None:
            predictions[target] = "⚠️ Model not loaded"
            continue

        try:
            pred = model.predict(X_new)[0]
            prob = model.predict_proba(X_new)[0, 1] if hasattr(model, "predict_proba") else None
            predictions[target] = f"Prediction: {pred}" + (f", Probability: {prob:.3f}" if prob is not None else "")
        except Exception as e:
            predictions[target] = f"⚠️ Error predicting: {e}"

    return html.Ul([html.Li(f"{t}: {v}") for t, v in predictions.items()])

# ---------------------------
# Run server
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    server.run(debug=True, port=port)
