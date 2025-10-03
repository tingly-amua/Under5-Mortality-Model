import os
import pickle
import gdown
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
from flask import Flask, jsonify, request
import dash_bootstrap_components as dbc

# ---------------------------
# Google Drive file IDs (UPDATED)
# ---------------------------
FEATURES_FILE_ID = "1LITbeocbOLTcZBmf0KeBTLcch_03oRi7"   # feature_importances.pkl
MODEL_FILE_ID = "19H7NxVfaAK0Ml23X9jfTewVvZjcuJVhq"    # final_model.pkl

FEATURES_PATH = "feature_importances.pkl"
MODEL_PATH = "final_model.pkl"

# ---------------------------
# Download files if not present (Heroku-safe)
# ---------------------------
for file_id, path in [(FEATURES_FILE_ID, FEATURES_PATH), (MODEL_FILE_ID, MODEL_PATH)]:
    if not os.path.exists(path):
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"⬇️ Downloading {path} from {url} ...")
        try:
            # Use fuzzy=True in case of Google Drive confirmation pages
            gdown.download(url, path, quiet=False, fuzzy=True)
        except Exception as e:
            print(f"❌ Failed to download {path}: {e}")

# ---------------------------
# Load features & models safely (with debug prints)
# ---------------------------
try:
    with open(MODEL_PATH, "rb") as f:
        trained_models = pickle.load(f)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Error loading models:", e)
    trained_models = {}

try:
    feature_importances = pd.read_pickle(FEATURES_PATH)
    if not isinstance(feature_importances, pd.DataFrame):
        raise ValueError("Pickled file is not a valid DataFrame")
    print("✅ Features loaded successfully")
    print("Available targets:", feature_importances['Target'].unique().tolist())
    print("Sample rows:\n", feature_importances.head())
except Exception as e:
    print("❌ Error loading feature importances:", e)
    feature_importances = pd.DataFrame(columns=['Target', 'Feature', 'Importance'])

# ---------------------------
# Extract top features dynamically
# ---------------------------
def get_top_features(target, top_n=20):
    if feature_importances.empty:
        print("⚠️ Feature importance DataFrame is empty")
        return []

    # Check required columns
    for col in ['Target', 'Feature', 'Importance']:
        if col not in feature_importances.columns:
            print(f"⚠️ Column '{col}' missing in feature_importances")
            return []

    filtered = feature_importances[
        feature_importances['Target'].str.lower() == target.lower()
    ]
    if filtered.empty:
        print(f"⚠️ No features found for target: '{target}'. Available targets: {feature_importances['Target'].unique()}")
        return []

    top_feats = filtered.nlargest(top_n, 'Importance')['Feature'].tolist()
    print(f"✅ Top {len(top_feats)} features for '{target}': {top_feats[:5]} …")
    return top_feats

def make_dropdown(target):
    return dcc.Dropdown(
        id=f'dropdown-{target.lower()}',
        options=[{'label': f, 'value': f} for f in get_top_features(target)],
        placeholder=f"Predictive features for {target}",
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
    prediction = trained_models.get("Under5", lambda x: "Model not loaded")(data)
    return jsonify({"prediction": prediction})

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
        dbc.Col(html.Button("Under5", id='btn-under5', n_clicks=0, className="btn btn-info"), width="auto"),
        dbc.Col(html.Button("Infant", id='btn-infant', n_clicks=0, className="btn btn-info"), width="auto"),
        dbc.Col(html.Button("Neonatal", id='btn-neonatal', n_clicks=0, className="btn btn-info"), width="auto"),
    ], justify="center", className="mb-4"),

    dbc.Row([
        dbc.Col(make_dropdown("Under5"), width=3),
        dbc.Col(make_dropdown("Infant"), width=3),
        dbc.Col(make_dropdown("Neonatal"), width=3),
    ], justify="center", className="mb-4"),

    dbc.Row([dbc.Col(html.Button("Predict", id='predict-btn', n_clicks=0, className="btn btn-success"), width="auto")],
            justify="center", className="mb-4"),

    dbc.Row([dbc.Col(html.Div(id='prediction-output', className="text-center"), width=12)])
], fluid=True)

@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-btn', 'n_clicks'),
    State('dropdown-under5', 'value'),
    State('dropdown-infant', 'value'),
    State('dropdown-neonatal', 'value')
)
def make_prediction(n_clicks, u5, inf, neo):
    if n_clicks < 1:
        return "ℹ️ Select features first."
    sel = {
        'Under5': u5 or [],
        'Infant': inf or [],
        'Neonatal': neo or []
    }
    return f"✅ Selected features: {sel}"

# ---------------------------
# Run server
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    server.run(debug=True, port=port)
