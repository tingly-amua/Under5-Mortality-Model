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
FEATURES_FILE_ID = "1Wp4NOmMveYMql2h8GVfyx0J8pmNU7pkQ"
MODEL_FILE_ID = "1sxZBckDWmumOd7Yilg4_0oudNlqLOWZu"

FEATURES_PATH = "feature_importances.pkl"
MODEL_PATH = "final_model.pkl"

# ---------------------------
# Download files if not present
# ---------------------------
for file_id, path in [(FEATURES_FILE_ID, FEATURES_PATH), (MODEL_FILE_ID, MODEL_PATH)]:
    if not os.path.exists(path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, path, quiet=False)

# ---------------------------
# Load features & models
# ---------------------------
try:
    with open(MODEL_PATH, "rb") as f:
        trained_models = pickle.load(f)
except Exception as e:
    print("❌ Error loading models:", e)
    trained_models = {}

try:
    feature_importances = pd.read_pickle(FEATURES_PATH)
except Exception as e:
    print("❌ Error loading feature importances:", e)
    feature_importances = pd.DataFrame()

# ---------------------------
# Extract top features dynamically
# ---------------------------
def get_top_features(target, top_n=20):
    """Return top N features for a given target from the feature importance DataFrame."""
    filtered = feature_importances[feature_importances['Target'] == target]
    top_features = filtered.nlargest(top_n, 'Importance')['Feature'].tolist()
    return top_features

top_features_under5 = get_top_features('Under5')
top_features_infant = get_top_features('Infant')
top_features_neonatal = get_top_features('Neonatal')

# ---------------------------
# Flask server
# ---------------------------
server = Flask(__name__)

# Landing page
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
        background: linear-gradient(to bottom, #a0e1fa, #d7f4fa);
        color: #004080;
    ">
        <h1 style='font-size: 4em;'>👶 Afya Toto</h1>
        <p style='font-size: 1.5em; max-width:600px;'>
        Protecting Children’s Health Through Data Insights
        </p>
        <img src='https://i.pinimg.com/1200x/97/a9/1c/97a91c944845237ef509452fec78863f.jpg'
             alt='Child Health' style='width:300px; margin:20px; border-radius:15px;'/>
        <p style='max-width:600px;'>Under-5 Mortality Rate: 45/1000 | SDG 3 Goal: Reduce Child Mortality</p>
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

# API route for programmatic prediction
@server.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.json
    # TODO: replace with actual model prediction logic
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

# ---------------------------
# Dash layout
# ---------------------------
app.layout = dbc.Container([
    dbc.Row([dbc.Col(html.H1("Afya-Toto Dashboard", className="text-center text-primary mb-4"), width=12)]),

    # Target variable buttons
    dbc.Row([
        dbc.Col(html.Button("Under5", id='btn-under5', n_clicks=0,
                            className="btn btn-info mx-2"), width="auto"),
        dbc.Col(html.Button("Infant", id='btn-infant', n_clicks=0,
                            className="btn btn-info mx-2"), width="auto"),
        dbc.Col(html.Button("Neonatal", id='btn-neonatal', n_clicks=0,
                            className="btn btn-info mx-2"), width="auto")
    ], justify="center", className="mb-4"),

    # Predictive feature dropdowns
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

    # Predict button
    dbc.Row([dbc.Col(html.Button("Predict", id='predict-btn', n_clicks=0,
                                 className="btn btn-success"), width="auto")],
            justify="center", className="mb-4"),

    # Prediction output
    dbc.Row([dbc.Col(html.Div(id='prediction-output', className="text-center"), width=12)])
], fluid=True)

# ---------------------------
# Callback for prediction
# ---------------------------
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
        'Under5': under5_features,
        'Infant': infant_features,
        'Neonatal': neonatal_features
    }
    # TODO: insert actual prediction logic using trained_models
    return f"✅ Selected features for prediction: {selected_features}"

# ---------------------------
# Run server
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    server.run(debug=True, port=port)
