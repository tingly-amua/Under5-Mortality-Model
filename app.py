import os
import pickle
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
from flask import Flask, jsonify, request
import dash_bootstrap_components as dbc

# ---------------------------
# --- File Paths (Local) ----
# ---------------------------
# The app now expects these files to be in the same directory.
# No more downloading from Google Drive.
FEATURES_PATH = "feature_importances.pkl"
MODEL_PATH = "final_model.pkl"

# ---------------------------
# --- Load features & models safely ---
# ---------------------------
try:
    with open(MODEL_PATH, "rb") as f:
        trained_models = pickle.load(f)
    print("✅ Models loaded successfully from local file.")
except Exception as e:
    print(f"❌ Error loading models from {MODEL_PATH}: {e}")
    # Fallback to an empty dictionary if loading fails
    trained_models = {}

try:
    feature_importances = pd.read_pickle(FEATURES_PATH)
    if not isinstance(feature_importances, pd.DataFrame):
        raise ValueError("Pickled file is not a valid DataFrame")
    print("✅ Feature importances loaded successfully from local file.")
except Exception as e:
    print(f"❌ Error loading feature importances from {FEATURES_PATH}: {e}")
    # Fallback to an empty DataFrame if loading fails
    feature_importances = pd.DataFrame(columns=['Target', 'Feature', 'Importance'])

# ---------------------------
# --- Extract top features dynamically (robust version) ---
# ---------------------------
def get_top_features(target, top_n=20):
    """
    Extracts the top N features for a given target from the feature_importances DataFrame.
    Includes robust checks for empty data or missing columns.
    """
    if feature_importances.empty:
        print("⚠️ Feature importance DataFrame is empty. Cannot extract features.")
        return []

    # Ensure all required columns exist in the DataFrame
    required_cols = ['Target', 'Feature', 'Importance']
    if not all(col in feature_importances.columns for col in required_cols):
        print(f"⚠️ DataFrame is missing one of the required columns: {required_cols}")
        return []

    # Filter features for the specified target (case-insensitive)
    filtered_df = feature_importances[
        feature_importances['Target'].str.lower() == target.lower()
    ]

    if filtered_df.empty:
        print(f"⚠️ No features found for target: '{target}'. Available targets: {feature_importances['Target'].unique()}")
        return []

    # Get the top N features based on 'Importance'
    top_features = filtered_df.nlargest(top_n, 'Importance')['Feature'].tolist()
    print(f"✅ Extracted top {len(top_features)} features for target '{target}'.")
    return top_features

# ---------------------------
# --- Load top features for each category ---
# ---------------------------
top_features_under5 = get_top_features('Under5')
top_features_infant = get_top_features('Infant')
top_features_neonatal = get_top_features('Neonatal')


# ---------------------------
# --- Flask Server Setup ---
# ---------------------------
server = Flask(__name__)

@server.route("/")
def index():
    """Landing page HTML."""
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
# --- API Endpoint for Predictions ---
# ---------------------------
@server.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint to get predictions from the model."""
    data = request.json
    # Example prediction logic for 'Under5' target
    prediction_model = trained_models.get("Under5")
    if prediction_model:
        # This part needs to be implemented based on your model's expected input
        prediction = "Prediction logic here" # e.g., prediction_model.predict(data)
    else:
        prediction = "Model for 'Under5' not found."
    return jsonify({"prediction": prediction})

# ---------------------------
# --- Dash App Definition ---
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
        ), width=4),
        dbc.Col(dcc.Dropdown(
            id='dropdown-infant',
            options=[{'label': f, 'value': f} for f in top_features_infant],
            placeholder="Predictive features for Infant", multi=True
        ), width=4),
        dbc.Col(dcc.Dropdown(
            id='dropdown-neonatal',
            options=[{'label': f, 'value': f} for f in top_features_neonatal],
            placeholder="Predictive features for Neonatal", multi=True
        ), width=4)
    ], justify="center", className="mb-4"),

    dbc.Row([dbc.Col(html.Button("Predict", id='predict-btn', n_clicks=0, className="btn btn-success"), width="auto")],
            justify="center", className="mb-4"),

    dbc.Row([dbc.Col(html.Div(id='prediction-output', className="text-center mt-3"), width=12)])
], fluid=True)

@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-btn', 'n_clicks'),
    [State('dropdown-under5', 'value'),
     State('dropdown-infant', 'value'),
     State('dropdown-neonatal', 'value')]
)
def make_prediction(n_clicks, under5_features, infant_features, neonatal_features):
    if n_clicks < 1:
        return "ℹ️ Select features from the dropdowns and click Predict."
    
    # Create a DataFrame or dictionary from selected features for the model
    # Note: This is a placeholder. You will need to adapt this to match
    # the exact input format your trained model expects.
    selected_features = {
        'Under5_Features': under5_features or [],
        'Infant_Features': infant_features or [],
        'Neonatal_Features': neonatal_features or []
    }
    
    # Placeholder for actual prediction logic
    # prediction_result = trained_models['Under5'].predict(input_data)
    
    return f"✅ Prediction logic would run with these features: {selected_features}"

# ---------------------------
# --- Run Server ---
# ---------------------------
if __name__ == "__main__":
    # Ensure the static directory exists if you have local assets like CSS or images
    os.makedirs("static", exist_ok=True)
    # Use the PORT environment variable provided by Heroku
    port = int(os.environ.get("PORT", 8050))
    # Set debug=False for production environments like Heroku
    server.run(host="0.0.0.0", port=port, debug=False)
