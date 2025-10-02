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
# Google Drive file IDs
# ---------------------------
FEATURES_FILE_ID = "13G-wF49ooTnQ3tfTvoCh9thy-DmBJj99"  # updated feature_importances.pkl
MODEL_FILE_ID = "1sxZBckDWmumOd7Yilg4_0oudNlqLOWZu"

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
            gdown.download(url, path, quiet=False)
        except Exception as e:
            print(f"❌ Failed to download {path}: {e}")

# ---------------------------
# Load features & models safely
# ---------------------------
try:
    with open(MODEL_PATH, "rb") as f:
        trained_models = pickle.load(f)
except Exception as e:
    print("❌ Error loading models:", e)
    trained_models = {}

try:
    feature_importances = pd.read_pickle(FEATURES_PATH)
    if not isinstance(feature_importances, pd.DataFrame):
        raise ValueError("Pickled file is not a valid DataFrame")
except Exception as e:
    print("❌ Error loading feature importances:", e)
    feature_importances = pd.DataFrame(columns=['Target', 'Feature', 'Importance'])

# ---------------------------
# Extract top features dynamically
# ---------------------------
def get_top_features(target, top_n=20):
    if feature_importances.empty:
        return []
    filtered = feature_importances[feature_importances['Target'] == target]
    if filtered.empty:
        return []
    return filtered.nlargest(top_n, 'Importance')['Feature'].tolist()

top_features_under5 = get_top_features('Under5')
top_features_infant = get_top_features('Infant')
top_features_neonatal = get_top_features('Neonatal')

# ---------------------------
# Flask server
# ---------------------------
server = Flask(__name__)

@server.route("/")
def index():
    return f"""
    <div style="
        text-align:center; 
        font-family:sans-serif; 
        min-height: 100vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        background: url('/static/background.jpg') no-repeat center center fixed;
        background-size: cover;
        color: #004080;
    ">
        <h1 style='font-size: 4em; background:rgba(255,255,255,0.7); padding:10px; border-radius:10px;'>👶 Afya Toto</h1>
        <p style='font-size: 1.5em; max-width:800px; background:rgba(255,255,255,0.7); padding:10px; border-radius:10px;'>
        Protecting Children’s Health Through Data Insights
        </p>
        <p style='max-width:800px; background:rgba(255,255,255,0.7); padding:10px; border-radius:10px;'>
        Under-5 Mortality Rate: 45/1000 | SDG 3 Goal: Reduce Child Mortality
        </p>
        <a href='/dashboard/' style='
            display:inline-block;
            margin-top:25px;
            padding:15px 30px;
            background:#007BFF;
            color:white;
            border-radius:10px;
            text-decoration:none;
            font-weight:bold;
            font-size: 18px;
        '>Go to Dashboard</a>
    </div>
    """

# ---------------------------
# API endpoint
# ---------------------------
@server.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.json
    prediction = trained_models.get("Under5", lambda x: "High Risk")(data)
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
    dbc.Row([dbc.Col(html.H1("Afya-Toto Dashboard", className="text-center text-primary mb-4"), width=12)]),

    dbc.Row([
        dbc.Col(html.Button("Under5", id='btn-under5', n_clicks=0, className="btn btn-info mx-2"), width="auto"),
        dbc.Col(html.Button("Infant", id='btn-infant', n_clicks=0, className="btn btn-info mx-2"), width="auto"),
        dbc.Col(html.Button("Neonatal", id='btn-neonatal', n_clicks=0, className="btn btn-info mx-2"), width="auto")
    ], justify="center", className="mb-4"),

    dbc.Row([
        dbc.Col(dcc.Dropdown(
            id='dropdown-under5',
            options=[{'label': f, 'value': f} for f in top_features_under5],
            placeholder="Predictive features for Under5", multi=True
        ), width=3),
        dbc.Col(dcc.Dropdown(
            id='dropdown-infant',
            options=[{'label': f, 'value': f} for f in top_features_infant],
            placeholder="Predictive features for Infant", multi=True
        ), width=3),
        dbc.Col(dcc.Dropdown(
            id='dropdown-neonatal',
            options=[{'label': f, 'value': f} for f in top_features_neonatal],
            placeholder="Predictive features for Neonatal", multi=True
        ), width=3)
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
def make_prediction(n_clicks, under5_features, infant_features, neonatal_features):
    if n_clicks < 1:
        return "ℹ️ Select features and click Predict."
    selected_features = {
        'Under5': under5_features or [],
        'Infant': infant_features or [],
        'Neonatal': neonatal_features or []
    }
    return f"✅ Selected features for prediction: {selected_features}"

# ---------------------------
# Run server
# ---------------------------
if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    port = int(os.environ.get("PORT", 8050))
    server.run(debug=True, port=port)
