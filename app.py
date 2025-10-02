import os
import pickle
import gdown
import dash
import pandas as pd
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from flask import Flask

# ---------------------------
# Google Drive file IDs
# ---------------------------
FEATURES_FILE_ID = "1Wp4NOmMveYMql2h8GVfyx0J8pmNU7pkQ"  # feature_importances.pkl
MODEL_FILE_ID = "1sxZBckDWmumOd7Yilg4_0oudNlqLOWZu"     # final_model.pkl

FEATURES_PATH = "feature_importances.pkl"
MODEL_PATH = "final_model.pkl"

# ---------------------------
# Download files if not present
# ---------------------------
for file_id, path in [(FEATURES_FILE_ID, FEATURES_PATH), (MODEL_FILE_ID, MODEL_PATH)]:
    if not os.path.exists(path):
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {path} from Google Drive...")
        gdown.download(url, path, quiet=False)

# ---------------------------
# Load features & models
# ---------------------------
try:
    with open(MODEL_PATH, "rb") as f:
        trained_models = pickle.load(f)  # dict: {'Under-5': model, 'Infant': model, 'Neonatal': model}
    print("✅ Models loaded.")
except Exception as e:
    print("❌ Error loading models:", e)
    trained_models = {}

try:
    feature_importances = pd.read_pickle(FEATURES_PATH)
    print("✅ Feature importances loaded.")
except Exception as e:
    print("❌ Error loading feature importances:", e)
    feature_importances = pd.DataFrame()

# Kenya-focused features (simplified for UI)
kenya_features = [
    'num__child_death_history', 'cat__Region_Kwale', 'num__Weight/Age standard deviation (new WHO)',
    'num__Childs height in centimeters (1 decimal)', 'cat__Region_Isiolo', 'cat__Region_Laikipia',
    'cat__Region_Mombasa', 'cat__Region_Nairobi', 'cat__Region_Baringo',
    'num__Childs weight in kilograms (1 decimal)'
]

# ---------------------------
# Flask server
# ---------------------------
server = Flask(__name__)

@server.route("/")
def index():
    return """
    <div style="
        text-align:center; 
        font-family:sans-serif; 
        height: 100vh;
        background-image: url('https://i.pinimg.com/1200x/97/a9/1c/97a91c944845237ef509452fec78863f.jpg');
        background-size: cover;
        background-position: center;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        color: white;
        text-shadow: 1px 1px 5px rgba(0,0,0,0.7);
    ">
        <h1 style='font-size: 4em;'>👶 Afya-Toto</h1>
        <p style='font-size: 1.5em;'>Under-5 Mortality Risk Prediction Tool - Kenya</p>
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
# Dash app
# ---------------------------
app = dash.Dash(
    __name__,
    server=server,
    url_base_pathname="/dashboard/",
    suppress_callback_exceptions=True
)

# Layout
app.layout = html.Div(
    style={"display": "flex", "minHeight": "100vh", "fontFamily": "sans-serif"},
    children=[
        # Sidebar
        html.Div(
            style={
                "flex": "1",
                "backgroundColor": "#e0f7fa",
                "padding": "30px",
                "borderRight": "2px solid #b2ebf2",
                "display": "flex",
                "flexDirection": "column",
                "justifyContent": "flex-start"
            },
            children=[
                html.H2("👶 Afya-Toto Inputs", style={"textAlign": "center", "color": "#007BFF", "marginBottom": "30px"}),

                html.Div(style={"marginBottom": "20px"}, children=[
                    html.P("Select feature(s):", style={"fontWeight": "bold"}),
                    dcc.Dropdown(
                        id="feature-dropdown",
                        options=[{"label": f, "value": f} for f in kenya_features],
                        placeholder="Select feature(s)",
                        multi=True,
                        searchable=True,
                    ),
                ]),

                html.Div(style={"marginBottom": "20px"}, children=[
                    html.P("Select target variable:", style={"fontWeight": "bold"}),
                    dcc.Dropdown(
                        id="target-dropdown",
                        options=[
                            {"label": "Under-5", "value": "Under-5"},
                            {"label": "Infant", "value": "Infant"},
                            {"label": "Neonatal", "value": "Neonatal"}
                        ],
                        placeholder="Select target variable",
                    ),
                ]),

                html.Button(
                    "👶 Predict",
                    id="predict-button",
                    n_clicks=0,
                    style={
                        "width": "100%",
                        "padding": "12px",
                        "backgroundColor": "#007BFF",
                        "color": "white",
                        "border": "none",
                        "borderRadius": "8px",
                        "fontSize": "16px",
                        "cursor": "pointer",
                        "marginTop": "10px"
                    },
                ),
            ],
        ),

        # Main panel
        html.Div(
            style={"flex": "3", "padding": "30px", "backgroundColor": "#f5f5f5"},
            children=[
                html.H2("📊 Prediction Results", style={"color": "#333", "marginBottom": "20px"}),

                html.Div(id="prediction-output", style={"fontSize": "20px", "marginTop": "20px"}),

                dcc.Loading(
                    id="loading-spinner",
                    type="circle",
                    children=dcc.Graph(id="prediction-chart"),
                ),
            ],
        ),
    ],
)

# ---------------------------
# Dash Callback
# ---------------------------
@app.callback(
    [Output("prediction-output", "children"),
     Output("prediction-chart", "figure")],
    [Input("predict-button", "n_clicks")],
    [State("feature-dropdown", "value"),
     State("target-dropdown", "value")]
)
def predict(n_clicks, features, target):
    if n_clicks < 1:
        return "ℹ️ Select inputs and click Predict.", go.Figure()

    if not target or target not in trained_models:
        return f"❌ Model for {target} not found.", go.Figure()

    try:
        model = trained_models[target]

        # For now, dummy input (all zeros). Later: replace with real user inputs.
        X_input = pd.DataFrame([[0] * len(features)], columns=features)

        pred = model.predict(X_input)[0]

        # Simple chart
        fig = go.Figure(go.Indicator(
            mode="number",
            value=pred,
            title={"text": f"Prediction for {target}"}
        ))

        return f"✅ Prediction for {target}: {pred}", fig

    except Exception as e:
        return f"❌ Error: {e}", go.Figure()

# ---------------------------
# Run server
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    server.run(debug=False, port=port, host="0.0.0.0")
