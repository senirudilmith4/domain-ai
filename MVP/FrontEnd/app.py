import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
from datetime import date, datetime, timedelta
import os
import numpy as np
import joblib
from db_manager import StudentDB, DB_PATH
import requests
import asyncio


# ── Import the separated PHI class ────────────────────────────────────────
from PHI_INT import GPAInterventionSystem

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & DARK THEME CSS
# -----------------------------------------------------------------------------
# ✅ ONLY ONE st.set_page_config() call — must be first Streamlit command
st.set_page_config(
    page_title="Domain Specific AI Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* ── Fonts ─────────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Figtree:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── CSS Variables ──────────────────────────────────────────────── */
    :root {
        --bg:        #000000;
        --surface:   #0a0a0a;
        --surface2:  #141414;
        --border:    rgba(255,255,255,0.08);
        --border-hi: rgba(0,210,180,0.3);
        --accent:    #00d2b4;
        --accent2:   #0071e3;
        --accent3:   #f5a623;
        --danger:    #ff3b30;
        --text:      #f5f5f7;
        --text-sub:  #86868b;
        --text-dim:  #424245;
        --glow:      0 0 40px rgba(0,210,180,0.1);
        --font:      'Figtree', -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif;
        --mono:      'JetBrains Mono', 'SF Mono', monospace;
    }

    /* ── Base ───────────────────────────────────────────────────────── */
    html, body, .stApp {
        background-color: var(--bg) !important;
        font-family: var(--font);
        color: var(--text);
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    /* ── Typography ─────────────────────────────────────────────────── */
    h1, h2, h3, h4 {
        font-family: var(--font) !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em !important;
        color: var(--text) !important;
        line-height: 1.15 !important;
    }

    /* ── Sidebar ────────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a0a 0%, #050505 100%) !important;
        border-right: 1px solid var(--border) !important;
        box-shadow: 4px 0 40px rgba(0,0,0,0.6);
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }

    /* Sidebar nav radio buttons */
    .stRadio > div {
        gap: 4px;
    }
    .stRadio label {
        background: transparent !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        transition: all 0.2s ease !important;
        border: 1px solid transparent !important;
        font-family: var(--font) !important;
        font-size: 0.9rem !important;
        color: var(--text-sub) !important;
        cursor: pointer !important;
    }
    .stRadio label:hover {
        background: rgba(0,210,180,0.07) !important;
        border-color: rgba(0,210,180,0.2) !important;
        color: var(--text) !important;
    }
    .stRadio label[data-checked="true"] {
        background: rgba(0,210,180,0.1) !important;
        border-color: var(--border-hi) !important;
        color: var(--accent) !important;
    }

    /* ── Cards ──────────────────────────────────────────────────────── */
    .css-card {
        background: linear-gradient(135deg, #141414 0%, #0a0a0a 100%);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04);
        transition: transform 0.25s cubic-bezier(.4,0,.2,1),
                    box-shadow 0.25s cubic-bezier(.4,0,.2,1),
                    border-color 0.25s ease;
        position: relative;
        overflow: hidden;
    }
    .css-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0,210,180,0.4), transparent);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    .css-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 60px rgba(0,0,0,0.5), var(--glow);
        border-color: var(--border-hi);
    }
    .css-card:hover::before {
        opacity: 1;
    }

    /* ── Metric tiles ───────────────────────────────────────────────── */
    .metric-value {
        font-family: var(--mono);
        font-size: 2rem;
        font-weight: 500;
        color: var(--accent);
        line-height: 1.1;
        letter-spacing: -0.03em;
    }
    .metric-label {
        color: var(--text-sub);
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .metric-sub {
        color: var(--text-dim);
        font-size: 0.75rem;
        margin-top: 4px;
    }

    /* ── Buttons ────────────────────────────────────────────────────── */
    div.stButton > button {
        background: linear-gradient(135deg, #00d2b4 0%, #0099ff 100%) !important;
        color: #07090f !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: var(--font) !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.04em !important;
        padding: 0.55rem 1.4rem !important;
        transition: all 0.2s cubic-bezier(.4,0,.2,1) !important;
        box-shadow: 0 4px 20px rgba(0,210,180,0.25) !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(0,210,180,0.4) !important;
        color: #07090f !important;
    }
    div.stButton > button:active {
        transform: translateY(0) !important;
    }

    /* ── Inputs & selects ───────────────────────────────────────────── */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div,
    .stDateInput > div > div > input {
        background-color: #111827 !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--text) !important;
        font-family: var(--font) !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: var(--border-hi) !important;
        box-shadow: 0 0 0 3px rgba(0,210,180,0.12) !important;
    }

    /* Slider */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
    }

    /* ── Chat ───────────────────────────────────────────────────────── */
    .stChatMessage {
        background: #111827 !important;
        border: 1px solid var(--border) !important;
        border-radius: 14px !important;
        padding: 1rem 1.2rem !important;
    }
    .stChatInputContainer {
        background: #0d1117 !important;
        border: 1px solid var(--border) !important;
        border-radius: 14px !important;
    }
    .stChatInputContainer > div > textarea {
        background: transparent !important;
        color: var(--text) !important;
        font-family: var(--font) !important;
    }

    /* ── Dataframe / table ──────────────────────────────────────────── */
    .stDataFrame {
        border: 1px solid var(--border) !important;
        border-radius: 14px !important;
        overflow: hidden !important;
    }

    /* ── Divider ────────────────────────────────────────────────────── */
    hr {
        border: none !important;
        border-top: 1px solid var(--border) !important;
        margin: 1.2rem 0 !important;
    }

    /* ── Status badges ──────────────────────────────────────────────── */
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        font-family: var(--mono);
    }
    .badge-critical { background: rgba(239,68,68,0.15);  color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
    .badge-high     { background: rgba(245,158,11,0.15); color: #fbbf24; border: 1px solid rgba(245,158,11,0.3); }
    .badge-medium   { background: rgba(59,130,246,0.15); color: #60a5fa; border: 1px solid rgba(59,130,246,0.3); }
    .badge-low      { background: rgba(0,210,180,0.12);  color: #00d2b4; border: 1px solid rgba(0,210,180,0.25); }

    /* ── Scrollbar ──────────────────────────────────────────────────── */
    ::-webkit-scrollbar       { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 999px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--accent); }

    /* ── Streamlit overrides ────────────────────────────────────────── */
    .stAlert {
        border-radius: 12px !important;
        border: 1px solid var(--border) !important;
    }
    .stSuccess { background: rgba(0,210,180,0.08) !important; border-color: rgba(0,210,180,0.3) !important; }
    .stWarning { background: rgba(245,158,11,0.08) !important; border-color: rgba(245,158,11,0.3) !important; }
    .stError   { background: rgba(239,68,68,0.08)  !important; border-color: rgba(239,68,68,0.3)  !important; }
    .stInfo    { background: rgba(59,130,246,0.08)  !important; border-color: rgba(59,130,246,0.3)  !important; }

    /* ── Caption & small text ───────────────────────────────────────── */
    .stCaption, small, .stMarkdown p:has(small) {
        color: var(--text-sub) !important;
        font-size: 0.78rem !important;
    }

    /* ── Form container ─────────────────────────────────────────────── */
    .stForm {
        background: #0d1117 !important;
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
        padding: 1.2rem !important;
    }

    /* ── Expander ───────────────────────────────────────────────────── */
    .streamlit-expanderHeader {
        background: #111827 !important;
        border-radius: 10px !important;
        font-family: var(--font) !important;
        font-weight: 600 !important;
    }

    /* ── Tabs ───────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 1px solid var(--border) !important;
        gap: 0 !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 8px 8px 0 0 !important;
        color: var(--text-sub) !important;
        font-family: var(--font) !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent) !important;
        border-bottom: 2px solid var(--accent) !important;
    }

    /* ── Multiselect tags ───────────────────────────────────────────── */
    .stMultiSelect span[data-baseweb="tag"] {
        background: rgba(0,210,180,0.15) !important;
        border: 1px solid rgba(0,210,180,0.3) !important;
        border-radius: 6px !important;
        color: var(--accent) !important;
    }

    /* ── Plotly chart bg ────────────────────────────────────────────── */
    .js-plotly-plot .plotly .bg {
        fill: transparent !important;
    }

    /* ── Spinner ────────────────────────────────────────────────────── */
    .stSpinner > div {
        border-top-color: var(--accent) !important;
    }

    /* ── Course Recommender result box ──────────────────────────────── */
    .result-box {
        padding: 20px;
        border-radius: 10px;
        background-color: rgba(20, 20, 20, 0.9);
        border-left: 5px solid #00d2b4;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.4);
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. SESSION STATE & MOCK DATA
# -----------------------------------------------------------------------------
if "login_state" not in st.session_state:
    st.session_state.login_state = False
if "user_role" not in st.session_state:
    st.session_state.user_role = "Student"
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# ✅ Single initialisation of messages — no duplicate block
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Welcome. I am your Domain Specific AI Assistant. I can help with University policies, Task Prioritization, and Course Recommendations."}
    ]

if "tasks_df" not in st.session_state:
    data = {
        "Task": ["Final Year Project Proposal", "Machine Learning Assignment 1", "Research Methodology Review",
                 "Course Registration"],
        "Module": ["PROJ400", "ML201", "RES301", "ADMIN"],
        "Deadline": [date.today() + timedelta(days=2), date.today() + timedelta(days=10),
                     date.today() + timedelta(days=5), date.today() + timedelta(days=1)],
        "Priority": ["Critical", "Medium", "High", "Low"],
        "Status": ["In Progress", "Not Started", "Not Started", "Done"],
        "Progress": [65, 0, 10, 100]
    }
    st.session_state.tasks_df = pd.DataFrame(data)

if "saved_courses" not in st.session_state:
    st.session_state.saved_courses = []


# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def stream_text(text):
    """Simulates AI typing effect."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.03)


def calculate_risk(df):
    """Simple algorithm to calculate workload risk."""
    pending = df[df['Status'] != 'Done']
    high_prio = len(pending[pending['Priority'].isin(['Critical', 'High'])])
    if high_prio > 2:
        return 85, "High Risk"
    elif high_prio > 0:
        return 50, "Moderate Risk"
    else:
        return 15, "Low Risk"


@st.cache_resource(show_spinner=False)
def load_priority_model():
    """Loads and caches the priority ML model to prevent redundant loading."""
    try:
        MODEL_PATH = "/Users/thushanthmahendran/Frontend/domain-ai/ml/TaskPriority/saved_models/best_model_pipeline.pkl"
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading Priority Model: {e}")
        return None


def is_valid_against_csv(new_row):
    """Validate new input against training CSV schema/range."""
    CSV_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'ml', 'TaskPriority', 'processed_task_data.csv')

    if not os.path.exists(CSV_PATH):
        return False, "Training CSV not found for validation."

    df_csv = pd.read_csv(CSV_PATH)

    required_cols = [
        "Year", "Semester", "Week_Released", "Week_Deadline", "Current_Week", "Weeks_Left",
        "Weight", "Difficulty", "Estimated_Hours", "Current_Workload", "Procrastination_Score",
        "Avg_Delay_History", "Urgency", "Task_Type_Exam", "Task_Type_Project", "Task_Type_Quiz", "Task_Type_Report"
    ]

    for col in required_cols:
        if col not in new_row:
            return False, f"Missing required field: {col}"

    checks = [
        ("Year", lambda v: int(v) in df_csv["Year"].unique()),
        ("Semester", lambda v: int(v) in df_csv["Semester"].unique()),
        ("Week_Released", lambda v: 1 <= int(v) <= 52),
        ("Week_Deadline", lambda v: 1 <= int(v) <= 52),
        ("Current_Week", lambda v: 1 <= int(v) <= 52),
        ("Weeks_Left", lambda v: 0 <= int(v) <= 52),
        ("Weight", lambda v: 0 <= int(v) <= 100),
        ("Difficulty", lambda v: 1.0 <= float(v) <= 10.0),
        ("Estimated_Hours", lambda v: 1 <= int(v) <= 40),
        ("Current_Workload", lambda v: 0.0 <= float(v) <= 20.0),
        ("Procrastination_Score", lambda v: 0.0 <= float(v) <= 1.0),
        ("Avg_Delay_History", lambda v: 0.0 <= float(v) <= 1.0),
        ("Urgency", lambda v: 0.0 <= float(v) <= 1.0),
        ("Task_Type_Exam", lambda v: v in [0, 1, True, False]),
        ("Task_Type_Project", lambda v: v in [0, 1, True, False]),
        ("Task_Type_Quiz", lambda v: v in [0, 1, True, False]),
        ("Task_Type_Report", lambda v: v in [0, 1, True, False]),
    ]

    for col, check in checks:
        try:
            if not check(new_row[col]):
                return False, f"{col} value '{new_row[col]}' is not valid based on CSV data."
        except Exception as exc:
            return False, f"{col} validation error: {exc}"

    return True, ""


def ai_prioritize(df):
    """Sorts dataframe and assigns priority using the trained Machine Learning Model."""
    if len(df[df['Status'] != 'Done']) > 0:
        model = load_priority_model()
        if model:
            def predict_priority(row: pd.Series) -> str:
                if str(row['Status']) == 'Done':
                    return "Low"

                deadline = pd.to_datetime(row['Deadline'])
                weeks_left = max(0, (deadline.date() - date.today()).days // 7)
                weight = 50
                urgency = round(min((1.0 / (1.0 + weeks_left)) * (1 + weight / 100.0), 1.0), 6) if weeks_left > 0 else 1.0

                task_name = str(row['Task']).lower()
                task_type = str(row.get('TaskType', '')).lower()

                features = {
                    "Weeks_Left": weeks_left, "Week_Released": 1, "Week_Deadline": weeks_left + 1,
                    "Current_Week": 2, "Semester": 1, "Year": 1,
                    "Weight": int(row.get('Weight', weight)),
                    "Difficulty": float(row.get('Difficulty', 5.0)),
                    "Estimated_Hours": int(row.get('EstHours', 8)),
                    "Current_Workload": 6.0,
                    "Procrastination_Score": 0.5, "Avg_Delay_History": 0.3,
                    "Urgency": urgency,
                    "Task_Type_Exam": 1 if 'exam' in task_name or 'exam' in task_type else 0,
                    "Task_Type_Project": 1 if 'project' in task_name or 'project' in task_type else 0,
                    "Task_Type_Quiz": 1 if 'quiz' in task_name or 'quiz' in task_type else 0,
                    "Task_Type_Report": 1 if 'review' in task_name or 'report' in task_name or 'report' in task_type else 0
                }

                cols = ["Weeks_Left", "Week_Released", "Week_Deadline", "Current_Week",
                        "Semester", "Year", "Weight", "Difficulty", "Estimated_Hours",
                        "Current_Workload", "Procrastination_Score", "Avg_Delay_History",
                        "Urgency", "Task_Type_Exam", "Task_Type_Project",
                        "Task_Type_Quiz", "Task_Type_Report"]

                X_vec = pd.DataFrame([features])[cols]
                pred_class = int(model.predict(X_vec)[0])

                if pred_class == 2:
                    return "Critical" if urgency > 0.8 else "High"
                return "Medium" if pred_class == 1 else "Low"

            df['Priority'] = df.apply(predict_priority, axis=1)

    prio_map = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    df['p_score'] = df['Priority'].map(prio_map)
    return df.sort_values(by=['p_score', 'Deadline']).drop(columns=['p_score'])


# -----------------------------------------------------------------------------
# 4. LOGIN SCREEN
# -----------------------------------------------------------------------------
if not st.session_state.login_state:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="
            background: linear-gradient(160deg, #0d1117 0%, #070b14 100%);
            padding: 48px 40px 36px;
            border-radius: 20px;
            border: 1px solid rgba(0,210,180,0.15);
            text-align: center;
            box-shadow: 0 30px 80px rgba(0,0,0,0.6), 0 0 60px rgba(0,210,180,0.05);
            position: relative;
            overflow: hidden;
        ">
            <div style="
                position: absolute; top: 0; left: 0; right: 0; height: 2px;
                background: linear-gradient(90deg, transparent, #00d2b4, #3b82f6, transparent);
            "></div>
            <div style="
                display: inline-flex; align-items: center; justify-content: center;
                width: 64px; height: 64px; border-radius: 16px;
                background: linear-gradient(135deg, rgba(0,210,180,0.15), rgba(59,130,246,0.15));
                border: 1px solid rgba(0,210,180,0.2);
                margin-bottom: 20px; font-size: 28px;
            ">🎓</div>
            <h2 style="
                font-family: 'Figtree', sans-serif; font-weight: 800; font-size: 1.6rem;
                background: linear-gradient(135deg, #00d2b4, #3b82f6);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                margin: 0 0 8px;
            ">Domain Specific AI Platform</h2>
            <p style="color: #64748b; font-size: 0.82rem; line-height: 1.6; margin: 0 0 24px;">
                Informatics Institute of Technology<br>
                <span style="color: #475569;">Robert Gordon University Aberdeen</span>
            </p>
            <div style="height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.06), transparent); margin-bottom: 24px;"></div>
        </div>
        """, unsafe_allow_html=True)

        st.write("**User Role:** Student")
        st.session_state.user_role = "Student"

        password = st.text_input("Password", type="password", placeholder="Enter your password...")

        if st.button("Authenticate", use_container_width=True):
            if password == "Student123":
                st.session_state.login_state = True
                st.rerun()
            elif password == "":
                st.warning("Please enter your password.")
            else:
                st.error("Incorrect password. Please check your credentials.")
    st.stop()

# -----------------------------------------------------------------------------
# 5. SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding: 0 4px 24px;">
        <div style="
            display: flex; align-items: center; gap: 10px;
            padding: 16px; border-radius: 14px;
            background: linear-gradient(135deg, rgba(0,210,180,0.08), rgba(59,130,246,0.06));
            border: 1px solid rgba(0,210,180,0.12);
            margin-bottom: 8px;
        ">
            <div style="
                width: 36px; height: 36px; border-radius: 10px; font-size: 18px;
                background: rgba(0,210,180,0.12); border: 1px solid rgba(0,210,180,0.2);
                display: flex; align-items: center; justify-content: center; flex-shrink: 0;
            ">🎓</div>
            <div>
                <div style="font-family: 'Figtree', sans-serif; font-weight: 700; font-size: 0.95rem; color: #f1f5f9; line-height: 1.2;">Domain Specific AI</div>
                <div style="font-size: 0.65rem; letter-spacing: 0.14em; text-transform: uppercase; color: #00d2b4; font-weight: 600;">University Platform</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="
        background: rgba(255,255,255,0.03); padding: 10px 14px;
        border-radius: 10px; margin-bottom: 16px;
        border: 1px solid rgba(255,255,255,0.05);
        display: flex; align-items: center; gap: 10px;
    ">
        <div style="width: 8px; height: 8px; border-radius: 50%; background: #00d2b4; box-shadow: 0 0 6px #00d2b4; flex-shrink:0;"></div>
        <div>
            <div style="font-size: 0.65rem; letter-spacing: 0.12em; text-transform: uppercase; color: #475569; margin-bottom: 1px;">Current Role</div>
            <div style="font-family: 'Figtree', sans-serif; font-weight: 700; font-size: 0.9rem; color: #f1f5f9;">{st.session_state.user_role}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    menu = st.radio(
        "MODULES",
        ["Dashboard", "Chat Assistant", "Task Manager", "GPA Predictor", "Course Recommender"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.caption("© 2025 Group 06 Project")

    if st.button("Log Out"):
        st.session_state.login_state = False
        st.rerun()

# -----------------------------------------------------------------------------
# 6. HEADER
# -----------------------------------------------------------------------------
st.markdown(f"""
<div style="
    display: flex; align-items: flex-end; justify-content: space-between;
    padding-bottom: 16px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 24px;
">
    <div>
        <div style="font-size: 0.65rem; letter-spacing: 0.18em; text-transform: uppercase; color: #00d2b4; font-weight: 600; margin-bottom: 4px;">Smart University AI</div>
        <h2 style="font-family: 'Figtree', sans-serif; font-weight: 800; font-size: 1.9rem; margin: 0; letter-spacing: -0.03em;">{menu}</h2>
        <p style="color: #475569; margin: 4px 0 0; font-size: 0.82rem;">
            Logged in as <span style="color: #64748b; font-weight: 600;">{st.session_state.user_role}</span>
            &nbsp;·&nbsp; Academic Year 2025/26
        </p>
    </div>
    <div style="text-align: right;">
        <div style="
            font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #334155;
            background: rgba(255,255,255,0.03); padding: 6px 12px;
            border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);
        ">{datetime.now().strftime("%a, %d %b %Y  %H:%M")}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 7. MODULE: DASHBOARD
# -----------------------------------------------------------------------------
if menu == "Dashboard":

    df = st.session_state.tasks_df
    risk_val, risk_label = calculate_risk(df)
    pending_df = df[df['Status'] != 'Done']
    done_df    = df[df['Status'] == 'Done']
    next_due   = pending_df['Deadline'].min() if not pending_df.empty else date.today()
    days_left  = (next_due - date.today()).days if not pending_df.empty else 0
    completion_pct = int(len(done_df) / len(df) * 100) if len(df) > 0 else 0

    def get_urgency(row):
        dl = pd.to_datetime(row['Deadline'])
        wl = max(0, (dl.date() - date.today()).days // 7)
        return round(min((1.0 / (1.0 + wl)) * 1.5, 1.0), 2)

    df['_urgency'] = df.apply(get_urgency, axis=1)

    prio_counts = df['Priority'].value_counts()
    critical_n  = prio_counts.get('Critical', 0)
    high_n      = prio_counts.get('High', 0)
    medium_n    = prio_counts.get('Medium', 0)
    low_n       = prio_counts.get('Low', 0)

    # ── Row 1: 4 metric tiles ────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)

    risk_color = '#ff3b30' if risk_val > 80 else '#f5a623' if risk_val > 40 else '#00d2b4'
    risk_grad  = 'linear-gradient(90deg,#ff3b30,#ff6b6b)' if risk_val > 80 else 'linear-gradient(90deg,#f5a623,#fbbf24)' if risk_val > 40 else 'linear-gradient(90deg,#00d2b4,#0071e3)'

    with m1:
        st.markdown(f"""
        <div class="css-card" style="padding:20px 22px;">
            <div class="metric-label">Workload Risk</div>
            <div class="metric-value" style="color:{risk_color}; font-size:1.55rem; margin:6px 0 8px;">{risk_label}</div>
            <div style="height:3px; border-radius:999px; background:rgba(255,255,255,0.04); overflow:hidden;">
                <div style="height:100%; width:{risk_val}%; border-radius:999px; background:{risk_grad};"></div>
            </div>
            <div style="font-size:0.7rem; color:#424245; margin-top:6px; font-weight:500;">AI Prediction Model</div>
        </div>""", unsafe_allow_html=True)

    with m2:
        deadline_color = '#ff3b30' if days_left <= 2 else '#f5a623' if days_left <= 5 else '#f5f5f7'
        st.markdown(f"""
        <div class="css-card" style="padding:20px 22px;">
            <div class="metric-label">Next Deadline</div>
            <div class="metric-value" style="color:{deadline_color}; font-size:1.55rem; margin:6px 0 4px;">{next_due.strftime('%d %b') if not pending_df.empty else '—'}</div>
            <div style="font-size:0.75rem; color:#86868b; margin-top:4px;">{f'{days_left}d remaining' if days_left >= 0 else 'Overdue'}</div>
        </div>""", unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="css-card" style="padding:20px 22px;">
            <div class="metric-label">Pending Tasks</div>
            <div class="metric-value" style="color:#f5f5f7; font-size:1.55rem; margin:6px 0 4px;">{len(pending_df)}</div>
            <div style="font-size:0.75rem; color:#86868b; margin-top:4px;">{len(done_df)} completed · {len(df)} total</div>
        </div>""", unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="css-card" style="padding:20px 22px;">
            <div class="metric-label">Overall Progress</div>
            <div class="metric-value" style="color:#00d2b4; font-size:1.55rem; margin:6px 0 8px;">{completion_pct}%</div>
            <div style="height:3px; border-radius:999px; background:rgba(255,255,255,0.04); overflow:hidden;">
                <div style="height:100%; width:{completion_pct}%; border-radius:999px; background:linear-gradient(90deg,#00d2b4,#0071e3);"></div>
            </div>
            <div style="font-size:0.7rem; color:#424245; margin-top:6px; font-weight:500;">Sprint Completion</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)

    # ── Row 2: Gauge + Priority donut + Urgency bars ─────────────────
    c1, c2, c3 = st.columns([1, 1, 2])

    with c1:
        st.markdown('<div class="css-card" style="padding:20px;">', unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.7rem; letter-spacing:0.1em; text-transform:uppercase; color:#86868b; font-weight:600; margin-bottom:2px;'>Stress Gauge</div>", unsafe_allow_html=True)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_val,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [None, 100], 'tickcolor': "#424245", 'tickwidth': 1, 'tickfont': {'color':'#424245','size':9}},
                'bar': {'color': risk_color, 'thickness': 0.25},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 0,
                'steps': [
                    {'range': [0,  50], 'color': "rgba(0,210,180,0.05)"},
                    {'range': [50, 80], 'color': "rgba(245,166,35,0.05)"},
                    {'range': [80,100], 'color': "rgba(255,59,48,0.07)"}],
            },
            number={'font': {'color': risk_color, 'family': 'Figtree', 'size': 28}, 'suffix': ''}
        ))
        fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "#86868b", 'family': "Figtree"},
                                height=200, margin=dict(l=16, r=16, t=16, b=0))
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="css-card" style="padding:20px;">', unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.7rem; letter-spacing:0.1em; text-transform:uppercase; color:#86868b; font-weight:600; margin-bottom:2px;'>Priority Split</div>", unsafe_allow_html=True)
        fig_pie = go.Figure(go.Pie(
            labels=['Critical','High','Medium','Low'],
            values=[max(critical_n,0.01), max(high_n,0.01), max(medium_n,0.01), max(low_n,0.01)],
            hole=0.68,
            marker=dict(colors=['#ff3b30','#f5a623','#0071e3','#00d2b4'],
                        line=dict(color='#000000', width=2)),
            textinfo='none',
            hovertemplate='<b>%{label}</b><br>%{value} tasks<extra></extra>'
        ))
        fig_pie.add_annotation(text=f"{len(df)}<br><span style='font-size:9px'>tasks</span>",
                               x=0.5, y=0.5, showarrow=False,
                               font=dict(size=18, color='#f5f5f7', family='Figtree'))
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
                              height=200, margin=dict(l=16, r=16, t=16, b=0))
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="css-card" style="padding:20px;">', unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.7rem; letter-spacing:0.1em; text-transform:uppercase; color:#86868b; font-weight:600; margin-bottom:12px;'>Task Urgency by Deadline</div>", unsafe_allow_html=True)
        plot_df = df[df['Status'] != 'Done'].sort_values('_urgency', ascending=False).head(6)
        if not plot_df.empty:
            pcolor_map = {'Critical': '#ff3b30','High': '#f5a623','Medium': '#0071e3','Low': '#00d2b4'}
            bar_colors = [pcolor_map.get(p,'#86868b') for p in plot_df['Priority']]
            fig_bar = go.Figure(go.Bar(
                x=plot_df['_urgency'],
                y=plot_df['Task'].str[:28],
                orientation='h',
                marker=dict(color=bar_colors, line=dict(width=0)),
                text=[f"{v:.2f}" for v in plot_df['_urgency']],
                textposition='outside',
                textfont=dict(color='#86868b', size=10, family='Figtree'),
                hovertemplate='<b>%{y}</b><br>Urgency: %{x:.2f}<extra></extra>'
            ))
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={'color': "#86868b", 'family': "Figtree"},
                xaxis=dict(range=[0,1.15], gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(0,0,0,0)", tickfont=dict(size=10)),
                yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=10, color='#86868b')),
                height=200, margin=dict(l=8, r=40, t=4, b=4),
                bargap=0.35
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.markdown("<p style='color:#86868b; font-size:0.85rem; padding:20px 0;'>All tasks completed 🎉</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 3: Priority breakdown legend + Task table preview ────────
    st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
    r1, r2 = st.columns([1, 2])

    with r1:
        st.markdown('''<div class="css-card" style="padding:20px;">''', unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.7rem; letter-spacing:0.1em; text-transform:uppercase; color:#86868b; font-weight:600; margin-bottom:14px;'>ML Priority Breakdown</div>", unsafe_allow_html=True)
        for label, count, color in [("Critical", critical_n, "#ff3b30"), ("High", high_n, "#f5a623"), ("Medium", medium_n, "#0071e3"), ("Low", low_n, "#00d2b4")]:
            pct = int(count / len(df) * 100) if len(df) > 0 else 0
            st.markdown(f"""
            <div style="margin-bottom:12px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                    <span style="font-size:0.82rem; font-weight:600; color:{color};">{label}</span>
                    <span style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#86868b;">{count} · {pct}%</span>
                </div>
                <div style="height:3px; border-radius:999px; background:rgba(255,255,255,0.04); overflow:hidden;">
                    <div style="height:100%; width:{pct}%; border-radius:999px; background:{color}; opacity:0.85;"></div>
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with r2:
        st.markdown('''<div class="css-card" style="padding:20px;">''', unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.7rem; letter-spacing:0.1em; text-transform:uppercase; color:#86868b; font-weight:600; margin-bottom:12px;'>Task Velocity This Week</div>", unsafe_allow_html=True)
        chart_data = pd.DataFrame({"Day": ["Mon", "Tue", "Wed", "Thu", "Fri"],
                                    "Completed": [2, 1, 3, 0, 4], "Added": [1, 2, 1, 5, 2]})
        fig_vel = px.bar(chart_data, x="Day", y=["Completed", "Added"],
                         barmode='group', color_discrete_sequence=["#00d2b4", "#0071e3"])
        fig_vel.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "#86868b", 'family': "Figtree"},
            xaxis=dict(gridcolor="rgba(255,255,255,0.03)", zerolinecolor="rgba(0,0,0,0)", tickfont=dict(size=10)),
            yaxis=dict(gridcolor="rgba(255,255,255,0.03)", zerolinecolor="rgba(0,0,0,0)", tickfont=dict(size=10)),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        font=dict(color="#86868b", size=10), bgcolor="rgba(0,0,0,0)"),
            height=200, margin=dict(l=4, r=4, t=24, b=4), bargap=0.3
        )
        st.plotly_chart(fig_vel, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 8. MODULE: CHAT ASSISTANT
# -----------------------------------------------------------------------------
elif menu == "Chat Assistant":

    from rag_client import send_rag_query_event, wait_for_run_output
    import nest_asyncio
    nest_asyncio.apply()

    def query_rag(user_input: str) -> str:
        """Send question to RAG pipeline via Inngest and wait for response."""
        try:
            event_id = asyncio.get_event_loop().run_until_complete(
                send_rag_query_event(question=user_input, top_k=5)
            )
            output = wait_for_run_output(event_id, timeout_s=120)
            return output.get("answer") or output.get("response") or str(output)
        except TimeoutError:
            return "⚠️ The RAG pipeline timed out. Please try again."
        except requests.exceptions.ConnectionError:
            return "⚠️ Cannot connect to Inngest. Make sure `inngest dev` is running."
        except Exception as e:
            return f"⚠️ Error: {str(e)}"

    def stream_chat_text(text: str, delay: float = 0.015):
        """Simulate streaming by yielding characters."""
        for char in text:
            yield char
            time.sleep(delay)

    def handle_response(user_input: str):
        """Query RAG, stream the reply, persist both messages."""
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching knowledge base..."):
                full_resp = query_rag(user_input)

            placeholder = st.empty()
            streamed = ""
            for chunk in stream_chat_text(full_resp):
                streamed += chunk
                placeholder.markdown(streamed + "▌")
            placeholder.markdown(streamed)

            st.session_state.messages.append({"role": "assistant", "content": streamed})

    col_chat, col_info = st.columns([3, 1])

    with col_info:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown("#### 💡 Quick Prompts")

        prompts = [
            "What are the learning outcomes of Programming fundamentals?",
            "Approved list of hospitals for mitigation form",
            "Difference between Programming Fundamentals and Object Oriented Programming?",
        ]

        for p in prompts:
            if st.button(p, key=f"qp_{p}", use_container_width=True):
                st.session_state.pending_prompt = p
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)
        st.info("🔗 Connected to RAG pipeline via Inngest (local dev)")

    with col_chat:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if st.session_state.pending_prompt:
            prompt = st.session_state.pending_prompt
            st.session_state.pending_prompt = None
            handle_response(prompt)

        if user_input := st.chat_input("Ask the Domain AI..."):
            handle_response(user_input)

# -----------------------------------------------------------------------------
# 9. MODULE: TASK MANAGER
# -----------------------------------------------------------------------------
elif menu == "Task Manager":

    now = datetime.now().strftime('%A, %d %B %Y  %I:%M:%S %p')
    st.markdown(
        f"<div style='font-family: \"JetBrains Mono\", monospace; font-size: 0.78rem; color: #334155; background: rgba(255,255,255,0.02); padding: 8px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.04); display:inline-block; margin-bottom: 14px;'>🕒  {now}</div>",
        unsafe_allow_html=True
    )

    col_ctrl, col_table = st.columns([1, 3])

    with col_ctrl:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown("#### ⚡ AI Controls")

        if st.button("🤖 Auto-Prioritize Tasks", use_container_width=True):
            with st.spinner("AI is evaluating deadlines & importance..."):
                time.sleep(1)
                st.session_state.tasks_df = ai_prioritize(st.session_state.tasks_df)
            st.success("Workload Optimized.")
            st.rerun()

        st.markdown("---")
        st.markdown("#### Add New Task")
        with st.form("new_task"):
            col_ys, col_w = st.columns([2, 1])
            with col_ys:
                t_year = st.selectbox("Academic Year", [1, 2, 3, 4], index=0)
                t_sem = st.selectbox("Semester", [1, 2], index=0)
            with col_w:
                t_week_released = st.number_input("Week Released", min_value=1, max_value=52, value=1, step=1)
                t_week_deadline = st.number_input("Week Deadline", min_value=t_week_released, max_value=52, value=t_week_released+1, step=1)
                t_current_week  = st.number_input("Current Week", min_value=t_week_released, max_value=52, value=t_week_released, step=1)
                t_weeks_left    = max(0, t_week_deadline - t_current_week)

            t_name = st.text_input("Task Description", placeholder="e.g. Final sprint backlog cleanup")
            t_mod  = st.text_input("Module Code", max_chars=10)
            t_date = st.date_input("Due Date", min_value=date.today())
            t_prio = st.selectbox("Priority", ["Critical", "High", "Medium", "Low"])

            col_a, col_b = st.columns(2)
            with col_a:
                t_type     = st.selectbox("Task Type", ["Project", "Exam", "Quiz", "Report", "Other"])
                t_diff     = st.slider("Estimated Difficulty", 1.0, 10.0, 5.0, 0.5)
                t_workload = st.slider("Current Workload (hrs)", 0.0, 20.0, 6.0, 0.5)
                t_procrast = st.slider("Procrastination Score", 0.0, 1.0, 0.5, 0.01)
            with col_b:
                t_weight   = st.slider("Weight (%)", 0, 100, 50, 1)
                t_hours    = st.slider("Est. Hours Required", 1, 40, 8, 1)
                t_delay    = st.slider("Avg Delay History", 0.0, 1.0, 0.3, 0.01)
                t_urgency  = st.slider("Urgency", 0.0, 1.0, 0.1, 0.01)

            task_type_exam    = int(t_type == "Exam")
            task_type_project = int(t_type == "Project")
            task_type_quiz    = int(t_type == "Quiz")
            task_type_report  = int(t_type == "Report")

            errors = []
            if st.form_submit_button("Add to Workflow"):
                if not t_name.strip():
                    errors.append("Task Description cannot be empty.")
                if not t_mod.strip() or not t_mod.strip().isalnum():
                    errors.append("Module Code must be alphanumeric and not empty.")
                if t_week_deadline < t_week_released:
                    errors.append("Week Deadline cannot be before Week Released.")
                if t_current_week < t_week_released or t_current_week > t_week_deadline:
                    errors.append("Current Week must be between Week Released and Week Deadline.")
                if t_weight < 0 or t_weight > 100:
                    errors.append("Weight must be between 0 and 100.")
                if t_diff < 1.0 or t_diff > 10.0:
                    errors.append("Difficulty must be between 1.0 and 10.0.")
                if t_hours < 1 or t_hours > 40:
                    errors.append("Estimated Hours must be between 1 and 40.")
                if t_workload < 0 or t_workload > 20:
                    errors.append("Current Workload must be between 0 and 20.")
                if not (0.0 <= t_procrast <= 1.0):
                    errors.append("Procrastination Score must be between 0.0 and 1.0.")
                if not (0.0 <= t_delay <= 1.0):
                    errors.append("Avg Delay History must be between 0.0 and 1.0.")
                if not (0.0 <= t_urgency <= 1.0):
                    errors.append("Urgency must be between 0.0 and 1.0.")

                if errors:
                    for err in errors:
                        st.error(err)
                else:
                    new_data = pd.DataFrame([{
                        "Task": t_name.strip(), "Module": t_mod.strip().upper(),
                        "Deadline": t_date, "Priority": t_prio,
                        "Status": "Not Started", "Progress": 0,
                        "Year": t_year, "Semester": t_sem,
                        "Week_Released": t_week_released, "Week_Deadline": t_week_deadline,
                        "Current_Week": t_current_week, "Weeks_Left": t_weeks_left,
                        "Weight": t_weight, "Difficulty": t_diff,
                        "Estimated_Hours": t_hours, "Current_Workload": t_workload,
                        "Procrastination_Score": t_procrast, "Avg_Delay_History": t_delay,
                        "Urgency": t_urgency,
                        "Task_Type_Exam": task_type_exam, "Task_Type_Project": task_type_project,
                        "Task_Type_Quiz": task_type_quiz, "Task_Type_Report": task_type_report
                    }])
                    st.session_state.tasks_df = pd.concat([st.session_state.tasks_df, new_data], ignore_index=True)
                    st.success("Task added successfully!")
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_table:
        st.markdown("### 📋 Active Workload")
        edited_df = st.data_editor(
            st.session_state.tasks_df,
            use_container_width=True,
            num_rows="dynamic",
            height=600,
            column_config={
                "Priority": st.column_config.SelectboxColumn(
                    "Priority", options=["Critical", "High", "Medium", "Low"],
                    width="small", required=True,
                ),
                "Status": st.column_config.SelectboxColumn(
                    "Status", options=["Not Started", "In Progress", "Done"], width="small",
                ),
                "Progress": st.column_config.ProgressColumn(
                    "Completion", min_value=0, max_value=100, format="%f%%",
                ),
                "Deadline": st.column_config.DateColumn("Due Date", format="DD MMM YYYY"),
            }
        )
        st.session_state.tasks_df = edited_df

# -----------------------------------------------------------------------------
# 10. MODULE: GPA PREDICTOR
# -----------------------------------------------------------------------------
elif menu == "GPA Predictor":

    # ✅ NO st.set_page_config() here — removed duplicate

    db = StudentDB(db_path=DB_PATH)
    MODEL_3YR_PATH = "3yrgpa_predictor_model.pkl"
    MODEL_4YR_PATH = "4yrgpa_predictor_model.pkl"
    DATA_PATH      = "final_dataset.csv"

    if not os.path.exists(MODEL_3YR_PATH) or not os.path.exists(MODEL_4YR_PATH):
        st.error("Error: Required model files not found.")
        st.stop()

    try:
        model_3yr = joblib.load(MODEL_3YR_PATH)
        model_4yr = joblib.load(MODEL_4YR_PATH)
        full_data = pd.read_csv(DATA_PATH) if os.path.exists(DATA_PATH) else None
        if full_data is None:
            st.warning(f"Dataset '{DATA_PATH}' not found. Using default threshold.")
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

    features_3yr = [
        'Stage', 'hours_per_week', 'academic_stress', 'struggle_with_managing',
        'start_assignments_closer_deadline', 'Skill_Programming', 'Skill_Math',
        'Skill_Technical_Comm', 'Skill_Web_Development',
        'Grade_CM1601', 'Grade_CM1602', 'Grade_CM1603', 'Grade_CM1604',
        'Grade_CM1605', 'Grade_CM1606', 'Grade_CM2601', 'Grade_CM2602',
        'Grade_CM2603', 'Grade_CM2604', 'Grade_CM2605', 'Grade_CM2606',
        'Grade_CM2607', 'Year1_GPA', 'Year2_GPA',
        'Balancing multiple courses or projects',
        'Not knowing how to prioritize task',
        'Starting tasks too late'
    ]

    features_4yr = [
        'Stage', 'hours_per_week', 'academic_stress', 'struggle_with_managing',
        'start_assignments_closer_deadline', 'Skill_Programming', 'Skill_Math',
        'Skill_Technical_Comm', 'Skill_Web_Development',
        'Grade_CM1601', 'Grade_CM1602', 'Grade_CM1603', 'Grade_CM1604',
        'Grade_CM1605', 'Grade_CM1606', 'Grade_CM2601', 'Grade_CM2602',
        'Grade_CM2603', 'Grade_CM2604', 'Grade_CM2605', 'Grade_CM2606',
        'Grade_CM2607', 'Grade_CM3606', 'Grade_CM3604', 'Grade_CM3602',
        'Grade_CM3603', 'Year1_GPA', 'Year2_GPA', 'Year3_GPA',
        'Balancing multiple courses or projects',
        'Not knowing how to prioritize task',
        'Starting tasks too late'
    ]

    grade_point_map     = {'A': 4.0, 'B': 3.0, 'C': 2.0, 'D': 1.0, 'NOT_TAKEN': 0.0}
    grade_point_options = sorted(list(grade_point_map.keys()),
                                 key=lambda x: grade_point_map[x], reverse=True)

    def get_user_inputs(target_year, current_features):
        st.sidebar.header("📊 Personal & Skill Inputs")
        st.sidebar.selectbox("Current Academic Stage",
                             [f'Stage {i}' for i in range(1, 5)],
                             index=target_year - 1, disabled=True)

        st.sidebar.subheader("Study & Stress Factors")
        hours_mapping = {
            'Less than 3hrs': 1.5, '3 - 5 hrs': 4.0,
            '5 - 10 hrs': 7.5, 'More than 10hrs': 12.5
        }
        hours_per_week = hours_mapping[
            st.sidebar.selectbox("Study Hours Per Week", list(hours_mapping.keys()), index=2)
        ]
        academic_stress = st.sidebar.slider("Academic Stress Level (1=Low, 5=High)", 1, 5, 3)

        st.sidebar.subheader("Time Management")
        struggle_mapping = {'Rarely': 1, 'Sometimes': 2, 'Often/Always': 3}
        struggle_with_managing = struggle_mapping[
            st.sidebar.selectbox("Struggle with managing academic tasks?",
                                 list(struggle_mapping.keys()), index=1)
        ]
        deadline_mapping = {'Rarely/Never': 1, 'Sometimes': 2, 'Always': 3}
        start_assignments_closer_deadline = deadline_mapping[
            st.sidebar.selectbox("Start assignments closer to the deadline?",
                                 list(deadline_mapping.keys()), index=1)
        ]

        st.sidebar.subheader("Skill Levels (1=Novice, 5=Expert)")
        skill_programming = st.sidebar.slider("Skill_Programming", 1, 5, 3)
        skill_math        = st.sidebar.slider("Skill_Math", 1, 5, 3)
        skill_tech_comm   = st.sidebar.slider("Skill_Technical_Comm", 1, 5, 3)
        skill_web_dev     = st.sidebar.slider("Skill_Web_Development", 1, 5, 3)

        st.sidebar.header("🤔 Workload Challenges")
        challenge_balance  = st.sidebar.checkbox('Balancing multiple courses or projects', value=True)
        challenge_priority = st.sidebar.checkbox('Not knowing how to prioritize task', value=False)
        challenge_starting = st.sidebar.checkbox('Starting tasks too late', value=False)

        st.sidebar.subheader("Previous GPA Inputs")
        year1_gpa = st.sidebar.number_input("Year 1 GPA", min_value=0.0, max_value=4.0, step=0.01, value=0.0)
        year2_gpa = st.sidebar.number_input("Year 2 GPA", min_value=0.0, max_value=4.0, step=0.01, value=0.0)
        year3_gpa = 0.0
        if target_year == 4:
            year3_gpa = st.sidebar.number_input("Year 3 GPA (if available)",
                                                min_value=0.0, max_value=4.0, step=0.01, value=0.0)

        st.header("🎓 Course Grades (Previous Stages)")
        st.markdown("Enter final grades for courses up to the target stage.")

        grade_cols = [col for col in current_features if col.startswith('Grade_CM')]
        cols = st.columns(3)
        grade_inputs = {}
        for i, col in enumerate(grade_cols):
            course_name   = col.replace('Grade_', '')
            default_grade = ('A' if 'CM16' in course_name
                             else 'B' if 'CM26' in course_name
                             else 'C' if 'CM36' in course_name
                             else 'NOT_TAKEN')
            with cols[i % 3]:
                selected_grade    = st.selectbox(f"{course_name} Grade:",
                                                 grade_point_options,
                                                 index=grade_point_options.index(default_grade),
                                                 key=col)
                grade_inputs[col] = grade_point_map[selected_grade]

        user_data = {
            'Stage': target_year, 'hours_per_week': hours_per_week,
            'academic_stress': academic_stress,
            'struggle_with_managing': struggle_with_managing,
            'start_assignments_closer_deadline': start_assignments_closer_deadline,
            'Skill_Programming': skill_programming, 'Skill_Math': skill_math,
            'Skill_Technical_Comm': skill_tech_comm, 'Skill_Web_Development': skill_web_dev,
            'Balancing multiple courses or projects': 1 if challenge_balance  else 0,
            'Not knowing how to prioritize task':     1 if challenge_priority else 0,
            'Starting tasks too late':                1 if challenge_starting else 0,
            'Year1_GPA': year1_gpa, 'Year2_GPA': year2_gpa, 'Year3_GPA': year3_gpa,
            **grade_inputs
        }

        final_data = {key: user_data[key] for key in current_features}
        input_df   = pd.DataFrame([final_data], columns=current_features)
        return input_df, user_data

    st.markdown("Get your GPA prediction **PLUS** personalized improvement strategies powered by ML")

    prediction_task = st.selectbox(
        "Select Target Prediction Year:",
        ["Predict 3rd Year GPA", "Predict 4th Year GPA"]
    )

    if "3rd Year" in prediction_task:
        target_year      = 3
        current_model    = model_3yr
        current_features = features_3yr
        st.warning("⚠️ Input all grades up to Stage 2 (CM26xx). Stage 3/4 courses should be 'NOT_TAKEN'.")
    else:
        target_year      = 4
        current_model    = model_4yr
        current_features = features_4yr
        st.warning("⚠️ Input all grades up to Stage 3 (CM36xx). Stage 4 courses should be 'NOT_TAKEN'.")

    intervention_system = GPAInterventionSystem(current_model, full_data, target_year, db=db)

    st.markdown("---")
    input_df, user_data_dict = get_user_inputs(target_year, current_features)

    if st.button("✨ Predict My GPA & Get Personalized Advice", type="primary", use_container_width=True):
        with st.spinner("Analysing your profile and generating recommendations..."):
            predicted_gpa = intervention_system.predict_gpa(input_df)
            db.save_prediction({"target_year": target_year,
                                "predicted_gpa": predicted_gpa,
                                "user_data": user_data_dict})

            current_gpa, phi_interventions = intervention_system.generate_interventions(
                user_data_dict, top_k=5
            )

            st.markdown("## Your GPA Prediction")
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                st.metric(label=f"Predicted Year {target_year} GPA", value=f"{predicted_gpa:.3f}")
            with col2:
                if predicted_gpa >= intervention_system.gpa_threshold:
                    st.success("✅ On Track")
                else:
                    st.warning("⚠️ Needs Attention")
            with col3:
                if predicted_gpa >= 3.7:
                    grade_label = "Excellent (A)"; st.balloons()
                elif predicted_gpa >= 3.0:
                    grade_label = "Good (B+/A-)"
                elif predicted_gpa >= 2.0:
                    grade_label = "Average (C/B)"
                else:
                    grade_label = "At Risk (< C)"
                st.info(f"📊 {grade_label}")

            if predicted_gpa >= 3.7:
                st.success("**Outstanding!** Keep up the great work!")
            elif predicted_gpa >= 3.0:
                st.info("**Very Good!** Check recommendations below for optimisation.")
            elif predicted_gpa >= 2.0:
                st.warning("**Good, but room to improve.** Focus on the priority interventions below.")
            else:
                st.error("**Needs Immediate Attention!** Please review all recommendations carefully.")

            st.markdown("---")

            if phi_interventions:
                st.markdown("## Your Personalized Improvement Plan")
                st.markdown(f"**Found {len(phi_interventions)} scientifically-proven intervention(s)**")

                for idx, intervention in enumerate(phi_interventions, 1):
                    fpp = intervention['fpp_score']
                    effectiveness = ("🟢 Very High Impact" if fpp >= 2.0
                                     else "🔵 High Impact"   if fpp >= 1.5
                                     else "🟡 Moderate Impact")

                    with st.expander(
                        f"**#{idx}: {intervention['icon']} {intervention['name']}** {effectiveness}",
                        expanded=(idx == 1)
                    ):
                        col_a, col_b = st.columns([2, 1])
                        with col_a:
                            st.markdown("**📋 What to do:**")
                            st.markdown(f"> {intervention['description']}")
                            st.markdown("**💡 Why this helps:**")
                            st.markdown(
                                f"> Increases your success odds by "
                                f"**{(fpp - 1) * 100:.0f}%** (FPP Score: {fpp:.2f})"
                            )
                        with col_b:
                            st.metric(label="Expected GPA Increase",
                                      value=f"+{intervention['gpa_improvement']:.3f}",
                                      delta=f"{intervention['modified_gpa']:.3f} (new GPA)")
                            st.markdown(f"**Domain:** {intervention['domain'].replace('_', ' ').title()}")
                            st.markdown(f"**FPP Score:** {fpp:.2f}")
            else:
                st.markdown("## ✅ Excellent Profile!")
                st.success("No critical interventions needed. Keep up the great work!")
                if predicted_gpa < intervention_system.gpa_threshold:
                    st.info("Focus on maintaining current good habits!")

            st.markdown("### Input Data")
            st.dataframe(input_df.T, use_container_width=True)

            st.markdown("---")
            st.markdown("### 💾 Export Your Results")
            col_e1, col_e2 = st.columns(2)

            with col_e1:
                summary_df = pd.DataFrame({
                    "Predicted GPA":             [f"{predicted_gpa:.3f}"],
                    "Risk Level":                ["On Track" if predicted_gpa >= intervention_system.gpa_threshold else "Needs Attention"],
                    "Number of Recommendations": [len(phi_interventions)],
                    "Top Intervention":          [phi_interventions[0]['name'] if phi_interventions else "None needed"],
                    "Expected Improvement":      [f"+{phi_interventions[0]['gpa_improvement']:.3f}" if phi_interventions else "N/A"],
                    "Analysis Date":             [datetime.now().strftime("%Y-%m-%d %H:%M")]
                })
                st.download_button("📄 Download Summary (CSV)",
                                   summary_df.to_csv(index=False),
                                   file_name=f"gpa_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                                   mime="text/csv")

            with col_e2:
                if phi_interventions:
                    interventions_df = pd.DataFrame({
                        "Priority":        [i + 1 for i in range(len(phi_interventions))],
                        "Intervention":    [i['name']               for i in phi_interventions],
                        "Domain":          [i['domain']             for i in phi_interventions],
                        "FPP Score":       [f"{i['fpp_score']:.2f}" for i in phi_interventions],
                        "GPA Improvement": [f"+{i['gpa_improvement']:.3f}" for i in phi_interventions],
                        "Description":     [i['description']        for i in phi_interventions]
                    })
                    st.download_button("📋 Download Full Plan (CSV)",
                                       interventions_df.to_csv(index=False),
                                       file_name=f"improvement_plan_{datetime.now().strftime('%Y%m%d')}.csv",
                                       mime="text/csv")

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Powered by Random Forest ML (R² > 0.95) + Probabilistic Helpful Interventions (PHI) Framework"
        "</div>",
        unsafe_allow_html=True
    )

# -----------------------------------------------------------------------------
# 11. MODULE: COURSE RECOMMENDER
# -----------------------------------------------------------------------------
elif menu == "Course Recommender":

    # ✅ NO st.set_page_config() here — removed duplicate
    # ✅ NO duplicate <style> block here — global CSS at top covers everything

    st.write("Personalized elective suggestions for Stage 3 and Stage 4 students.")

    @st.cache_resource
    def load_assets():
        try:
            knn3    = joblib.load('Stage3_elective_recommender_knn.pkl')
            knn4    = joblib.load('Stage4_elective_recommender_knn.pkl')
            scaler3 = joblib.load('gpa_scaler.pkl')
            scaler4 = joblib.load('gpa_scaler4.pkl')
            dataset = pd.read_csv('final_dataset.csv')
            return knn3, knn4, scaler3, scaler4, dataset
        except Exception as e:
            st.error(f"⚠️ Error loading files: {e}")
            return None, None, None, None, None

    knn3, knn4, scaler3, scaler4, df = load_assets()

    st.subheader("Student Academic Profile")
    selected_stage = st.selectbox("Current Academic Stage", options=[3, 4], index=0)

    col1, col2, col3 = st.columns(3)
    with col1:
        y1_gpa = st.number_input("Year 1 GPA", 0.0, 4.0, 3.0, 0.01)
    with col2:
        y2_gpa = st.number_input("Year 2 GPA", 0.0, 4.0, 3.0, 0.01)

    y3_gpa = 0.0
    if selected_stage == 4:
        with col3:
            y3_gpa = st.number_input("Year 3 GPA", 0.0, 4.0, 3.0, 0.01)

    if knn3 is not None and knn4 is not None:
        if st.button("Generate Recommendation"):
            with st.spinner('Analyzing peer patterns...'):
                try:
                    if selected_stage == 3:
                        X_input  = np.array([[3, y1_gpa, y2_gpa]])
                        X_scaled = scaler3.transform(X_input)
                        distances, indices = knn3.kneighbors(X_scaled)
                        neighbors = df.iloc[indices[0]]
                        avg_3602  = neighbors[neighbors['Grade_CM3602'] > 0]['Grade_CM3602'].mean()
                        avg_3603  = neighbors[neighbors['Grade_CM3603'] > 0]['Grade_CM3603'].mean()
                        if (avg_3602 or 0) > (avg_3603 or 0):
                            rec_course, score = "CM3602: Internet of Things (IoT)", avg_3602
                        else:
                            rec_course, score = "CM3603: Edge Artificial Intelligence", avg_3603
                    else:
                        X_input  = np.array([[4, y1_gpa, y2_gpa, y3_gpa]])
                        X_scaled = scaler4.transform(X_input)
                        distances, indices = knn4.kneighbors(X_scaled)
                        neighbors = df.iloc[indices[0]]
                        avg_4606  = neighbors[neighbors['Grade_CM4606'] > 0]['Grade_CM4606'].mean()
                        avg_4603  = neighbors[neighbors['Grade_CM4603'] > 0]['Grade_CM4603'].mean()
                        if (avg_4606 or 0) > (avg_4603 or 0):
                            rec_course, score = "CM4606: Machine Vision", avg_4606
                        else:
                            rec_course, score = "CM4603: Natural Language Processing", avg_4603

                    st.markdown(f"""
                    <div class="result-box">
                        <h3 style='color: #00d2b4;'>Top Pick: {rec_course}</h3>
                        <p style='color: #86868b;'>Students with a similar GPA profile to yours performed best in this elective.</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.metric(label="Predicted Success Score", value=f"{score:.2f} / 4.00")
                    st.progress(min(float(score / 4.0), 1.0) if not pd.isna(score) else 0.0)

                except Exception as e:
                    st.error(f"Error during recommendation: {e}")
    else:
        st.warning("Ensure all model files and the dataset are present in the directory.")

    st.divider()
    st.caption("Domain-Specific AI Assistant Platform | IIT Project")