import os
import pickle
import gdown  # for downloading from Google Drive
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from flask import Flask, request, jsonify

# ---------------------------
# Google Drive file IDs
# ---------------------------
FEATURES_FILE_ID = "1Wp4NOmMveYMql2h8GVfyx0J8pmNU7pkQ"  # feature_importances.pkl
MODEL_FILE_ID = "1sxZBckDWmumOd7Yilg4_0oudNlqLOWZu"     # final_model.pkl

FEATURES_PATH = "feature_importances.pkl"
MODEL_PATH = "final_model.pkl"

# ---------------------------
# Download from Google Drive if not present
# ---------------------------
for file_id, path in [(FEATURES_FILE_ID, FEATURES_PATH), (MODEL_FILE_ID, MODEL_PATH)]:
    if not os.path.exists(path):
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {path} from Google Drive...")
        gdown.download(url, path, quiet=False)

# ---------------------------
# Load feature importances and model
# ---------------------------
try:
    with open(FEATURES_PATH, "rb") as f:
        feature_importances = pickle.load(f)
    print("✅ Feature importances loaded.")
except Exception as e:
    print("❌ Error loading feature importances:", e)
    feature_importances = []

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("✅ Model loaded.")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None

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
        background-image: url('https://via.placeholder.com/1200x800.png?text=Children+Background');
        background-size: cover;
        background-position: center;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        color: white;
        text-shadow: 1px 1px 4px rgba(0,0,0,0.7);
    ">
        <h1>👶 Afya-Toto</h1>
        <p>Under-5 Mortality Risk Prediction Tool</p>
        <a href='/dashboard/' style='
            display:inline-block;
            margin-top:20px;
            padding:10px 20px;
            background:#007BFF;
            color:white;
            border-radius:5px;
            text-decoration:none;
            font-weight:bold;
        '>Go to Dashboard</a>
    </div>
    """

# ---------------------------
# Dash app for dashboard
# ---------------------------
app = dash.Dash(
    __name__, 
    server=server, 
    url_base_pathname="/dashboard/",
    suppress_callback_exceptions=True
)

# Layout
app.layout = html.Div(
    style={"display": "flex", "flexDirection": "row", "minHeight": "100vh"},
    children=[
        # Sidebar
        html.Div(
            style={
                "flex": "1",
                "backgroundColor": "#e9f7f5",
                "padding": "20px",
                "borderRight": "2px solid #ccc"
            },
            children=[
                html.H2("👶 Afya-Toto Inputs", style={"textAlign": "center", "color": "#007BFF"}),

                html.P("Select feature:", style={"fontWeight": "bold"}),
                dcc.Dropdown(
                    id="feature-dropdown",
                    options=[{"label": f, "value": f} for f in feature_importances],
                    placeholder="Select a feature",
                    multi=True,
                    style={"marginBottom": "20px"},
                ),

                html.P("Select target:", style={"fontWeight": "bold"}),
                dcc.Dropdown(
                    id="target-dropdown",
                    options=[{"label": "Under-5", "value": "under5"},
                             {"label": "Infant", "value": "infant"},
                             {"label": "Neonatal", "value": "neonatal"}],
                    placeholder="Select target variable",
                    style={"marginBottom": "20px"},
                ),

                html.Button(
                    "👶 Predict",
                    id="predict-button",
                    n_clicks=0,
                    style={
                        "width": "100%",
                        "padding": "10px",
                        "backgroundColor": "#007BFF",
                        "color": "white",
                        "border": "none",
                        "borderRadius": "5px",
                        "fontSize": "16px"
                    },
                ),
            ],
        ),

        # Main panel
        html.Div(
            style={"flex": "3", "padding": "20px", "backgroundColor": "#f9f9f9"},
            children=[
                html.H2("📊 Prediction Results", style={"color": "#333"}),
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
    Output("prediction-chart", "figure"),
    [Input("predict-button", "n_clicks")],
    [State("feature-dropdown", "value"), State("target-dropdown", "value")],
)
def update_chart(n_clicks, features_selected, target_value):
    if n_clicks == 0 or not features_selected or not target_value:
        return go.Figure().update_layout(
            title_text="Select features + target, then click Predict",
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[{
                "text": "Waiting for input...",
                "xref": "paper", "yref": "paper",
                "showarrow": False, "font": {"size": 16, "color": "#888"}
            }],
        )

    try:
        # Here we assume features_selected is already ordered correctly for the model
        pred = model.predict([features_selected]).tolist()[0]
    except Exception as e:
        return go.Figure().update_layout(
            title_text="Error",
            annotations=[{
                "text": str(e),
                "xref": "paper", "yref": "paper",
                "showarrow": False,
                "font": {"size": 14, "color": "red"}
            }],
        )

    color = "red" if pred > 0.5 else "green"
    icon = "⚠️" if pred > 0.5 else "✅"

    fig = go.Figure(data=[go.Bar(
        x=[f"{icon} {target_value}"],
        y=[pred],
        marker_color=color,
        text=[f"{pred:.2f}"],
        textposition="auto"
    )])

    fig.update_layout(
        title_text=f"Prediction for {target_value}",
        yaxis_title="Predicted Risk",
        plot_bgcolor="white",
        paper_bgcolor="#f9f9f9",
        font={"color": "#333"},
    )
    return fig

# ---------------------------
# Run server
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    server.run(debug=False, port=port, host="0.0.0.0")
