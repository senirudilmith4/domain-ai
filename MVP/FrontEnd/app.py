import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
from datetime import date, datetime, timedelta
import os
import numpy as np

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & DARK THEME CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Domain Specific AI Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# COLOR PALETTE (Dark Mode)
# Background: #0E1117 (Streamlit Dark)
# Card BG: #1E1E1E
# Accent: #00ADB5 (Cyber Cyan)
# Text: #FAFAFA

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
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. SESSION STATE & MOCK DATA
# -----------------------------------------------------------------------------
if "login_state" not in st.session_state:
    st.session_state.login_state = False
if "user_role" not in st.session_state:
    st.session_state.user_role = "Student"

# Initial Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Welcome. I am your Domain Specific AI Assistant. I can help with University policies, Task Prioritization, and Course Recommendations."}
    ]

# Initial Tasks
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

    # Simple risk logic
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
        import joblib
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
                if str(row['Status']) == 'Done': return "Low"
                
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
# 4. LOGIN SCREEN (Simulated)
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
                font-family: var(--font); font-weight: 800; font-size: 1.6rem;
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

        # Only Student role is allowed for this app
        st.write("**User Role:** Student")
        st.session_state.user_role = "Student"

        password = st.text_input("Password", type="password", placeholder="Enter your password...")

        if st.button("Authenticate", width='stretch'):
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
                <div style="font-family: var(--font); font-weight: 700; font-size: 0.95rem; color: #f1f5f9; line-height: 1.2;">Domain Specific AI</div>
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
            <div style="font-family: var(--font); font-weight: 700; font-size: 0.9rem; color: #f1f5f9;">{st.session_state.user_role}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    menu = st.radio(
        "MODULES",
        ["Dashboard", "Chat Assistant", "Task Manager", "Course Recommender"],
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
        <h2 style="font-family: var(--font); font-weight: 800; font-size: 1.9rem; margin: 0; letter-spacing: -0.03em;">{menu}</h2>
        <p style="color: #475569; margin: 4px 0 0; font-size: 0.82rem;">
            Logged in as <span style="color: #64748b; font-weight: 600;">{st.session_state.user_role}</span>
            &nbsp;·&nbsp; Academic Year 2025/26
        </p>
    </div>
    <div style="text-align: right;">
        <div style="
            font-family: var(--mono); font-size: 0.75rem; color: #334155;
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

    # Compute per-task urgency scores for the priority breakdown
    def get_urgency(row):
        dl = pd.to_datetime(row['Deadline'])
        wl = max(0, (dl.date() - date.today()).days // 7)
        return round(min((1.0 / (1.0 + wl)) * 1.5, 1.0), 2)

    df['_urgency'] = df.apply(get_urgency, axis=1)

    # Priority breakdown counts
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
            <div style="font-size:0.7rem; color:var(--text-dim); margin-top:6px; font-weight:500;">AI Prediction Model</div>
        </div>""", unsafe_allow_html=True)

    with m2:
        deadline_color = '#ff3b30' if days_left <= 2 else '#f5a623' if days_left <= 5 else '#f5f5f7'
        st.markdown(f"""
        <div class="css-card" style="padding:20px 22px;">
            <div class="metric-label">Next Deadline</div>
            <div class="metric-value" style="color:{deadline_color}; font-size:1.55rem; margin:6px 0 4px;">{next_due.strftime('%d %b') if not pending_df.empty else '—'}</div>
            <div style="font-size:0.75rem; color:var(--text-sub); margin-top:4px;">{f'{days_left}d remaining' if days_left >= 0 else 'Overdue'}</div>
        </div>""", unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="css-card" style="padding:20px 22px;">
            <div class="metric-label">Pending Tasks</div>
            <div class="metric-value" style="color:var(--text); font-size:1.55rem; margin:6px 0 4px;">{len(pending_df)}</div>
            <div style="font-size:0.75rem; color:var(--text-sub); margin-top:4px;">{len(done_df)} completed · {len(df)} total</div>
        </div>""", unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="css-card" style="padding:20px 22px;">
            <div class="metric-label">Overall Progress</div>
            <div class="metric-value" style="color:#00d2b4; font-size:1.55rem; margin:6px 0 8px;">{completion_pct}%</div>
            <div style="height:3px; border-radius:999px; background:rgba(255,255,255,0.04); overflow:hidden;">
                <div style="height:100%; width:{completion_pct}%; border-radius:999px; background:linear-gradient(90deg,#00d2b4,#0071e3);"></div>
            </div>
            <div style="font-size:0.7rem; color:var(--text-dim); margin-top:6px; font-weight:500;">Sprint Completion</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)

    # ── Row 2: Gauge + Priority donut + Urgency bars ─────────────────
    c1, c2, c3 = st.columns([1, 1, 2])

    with c1:
        st.markdown('<div class="css-card" style="padding:20px;">', unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.7rem; letter-spacing:0.1em; text-transform:uppercase; color:var(--text-sub); font-weight:600; margin-bottom:2px;'>Stress Gauge</div>", unsafe_allow_html=True)
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
        st.markdown("<div style='font-size:0.7rem; letter-spacing:0.1em; text-transform:uppercase; color:var(--text-sub); font-weight:600; margin-bottom:2px;'>Priority Split</div>", unsafe_allow_html=True)
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
        st.markdown("<div style='font-size:0.7rem; letter-spacing:0.1em; text-transform:uppercase; color:var(--text-sub); font-weight:600; margin-bottom:12px;'>Task Urgency by Deadline</div>", unsafe_allow_html=True)
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
            st.markdown("<p style='color:var(--text-sub); font-size:0.85rem; padding:20px 0;'>All tasks completed 🎉</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 3: Priority breakdown legend + Task table preview ────────
    st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
    r1, r2 = st.columns([1, 2])

    with r1:
        st.markdown('''<div class="css-card" style="padding:20px;">''', unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.7rem; letter-spacing:0.1em; text-transform:uppercase; color:var(--text-sub); font-weight:600; margin-bottom:14px;'>ML Priority Breakdown</div>", unsafe_allow_html=True)
        for label, count, color in [("Critical", critical_n, "#ff3b30"), ("High", high_n, "#f5a623"), ("Medium", medium_n, "#0071e3"), ("Low", low_n, "#00d2b4")]:
            pct = int(count / len(df) * 100) if len(df) > 0 else 0
            st.markdown(f"""
            <div style="margin-bottom:12px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                    <span style="font-size:0.82rem; font-weight:600; color:{color};">{label}</span>
                    <span style="font-family:var(--mono); font-size:0.75rem; color:var(--text-sub);">{count} · {pct}%</span>
                </div>
                <div style="height:3px; border-radius:999px; background:rgba(255,255,255,0.04); overflow:hidden;">
                    <div style="height:100%; width:{pct}%; border-radius:999px; background:{color}; opacity:0.85;"></div>
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with r2:
        st.markdown('''<div class="css-card" style="padding:20px;">''', unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.7rem; letter-spacing:0.1em; text-transform:uppercase; color:var(--text-sub); font-weight:600; margin-bottom:12px;'>Task Velocity This Week</div>", unsafe_allow_html=True)
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

    col_chat, col_info = st.columns([3, 1])

    with col_info:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown("#### 💡 Quick Prompts")
        prompts = [
            "When is the proposal due?",
            "What is the policy on plagiarism?",
            "Draft an email to my supervisor.",
            "Summarize my workload."
        ]
        for p in prompts:
            if st.button(p, key=p, use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": p})
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.info("System connected to University Knowledge Base (Mock)")

    with col_chat:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if user_input := st.chat_input("Ask the Domain AI..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                # Mock Logic
                response = "I am analyzing the university database..."
                if "due" in user_input or "deadline" in user_input:
                    response = "Based on your Task Manager, the **Final Year Project Proposal** is your most critical deadline, due in 2 days. The system recommends dedicating 4 hours today."
                elif "policy" in user_input:
                    response = "According to the *Academic Integrity Policy 2025*, plagiarism is a Level 1 offense. Turnitin reports must be below 20% similarity."

                # Stream Response
                placeholder = st.empty()
                full_resp = ""
                for chunk in stream_text(response):
                    full_resp += chunk
                    placeholder.markdown(full_resp + "▌")
                placeholder.markdown(full_resp)
                st.session_state.messages.append({"role": "assistant", "content": full_resp})

# -----------------------------------------------------------------------------
# 9. MODULE: TASK MANAGER
# -----------------------------------------------------------------------------
elif menu == "Task Manager":

    # Display current date/time in the task manager header as polished card
    now = datetime.now().strftime('%A, %d %B %Y  %I:%M:%S %p')
    st.markdown(
        f"<div style='font-family: var(--mono); font-size: 0.78rem; color: #334155; background: rgba(255,255,255,0.02); padding: 8px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.04); display:inline-block; margin-bottom: 14px;'>🕒  {now}</div>",
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
            # Academic year and semester
            col_ys, col_w = st.columns([2, 1])
            with col_ys:
                t_year = st.selectbox("Academic Year", [1, 2, 3, 4], index=0, help="Year of study")
                t_sem = st.selectbox("Semester", [1, 2], index=0, help="Semester number")
            with col_w:
                t_week_released = st.number_input("Week Released", min_value=1, max_value=52, value=1, step=1, help="Week when task is released")
                t_week_deadline = st.number_input("Week Deadline", min_value=t_week_released, max_value=52, value=t_week_released+1, step=1, help="Week when task is due")
                t_current_week = st.number_input("Current Week", min_value=t_week_released, max_value=52, value=t_week_released, step=1, help="Current week number")
                t_weeks_left = max(0, t_week_deadline - t_current_week)

            t_name = st.text_input("Task Description", placeholder="e.g. Final sprint backlog cleanup", help="Short description of the task")
            t_mod = st.text_input("Module Code", max_chars=10, help="E.g. ML201, PROJ400")
            t_date = st.date_input("Due Date", min_value=date.today(), help="Task deadline date")
            t_prio = st.selectbox("Priority", ["Critical", "High", "Medium", "Low"])

            col_a, col_b = st.columns(2)
            with col_a:
                t_type = st.selectbox("Task Type", ["Project", "Exam", "Quiz", "Report", "Other"])
                t_diff = st.slider("Estimated Difficulty", 1.0, 10.0, 5.0, 0.5, help="1=Very Easy, 10=Very Hard")
                t_workload = st.slider("Current Workload (hrs)", 0.0, 20.0, 6.0, 0.5, help="Your current weekly workload")
                t_procrast = st.slider("Procrastination Score", 0.0, 1.0, 0.5, 0.01, help="0=Never, 1=Always")
            with col_b:
                t_weight = st.slider("Weight (%)", 0, 100, 50, 1, help="Contribution to final grade")
                t_hours = st.slider("Est. Hours Required", 1, 40, 8, 1, help="Estimated hours to complete")
                t_delay = st.slider("Avg Delay History", 0.0, 1.0, 0.3, 0.01, help="Average delay in past tasks")
                t_urgency = st.slider("Urgency", 0.0, 1.0, 0.1, 0.01, help="System-calculated or user estimate")

            # Task type one-hot encoding
            task_type_exam = int(t_type == "Exam")
            task_type_project = int(t_type == "Project")
            task_type_quiz = int(t_type == "Quiz")
            task_type_report = int(t_type == "Report")

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
                    # CSV-based validation
                    new_row = {
                        "Year": t_year,
                        "Semester": t_sem,
                        "Week_Released": t_week_released,
                        "Week_Deadline": t_week_deadline,
                        "Current_Week": t_current_week,
                        "Weeks_Left": t_weeks_left,
                        "Weight": t_weight,
                        "Difficulty": t_diff,
                        "Estimated_Hours": t_hours,
                        "Current_Workload": t_workload,
                        "Procrastination_Score": t_procrast,
                        "Avg_Delay_History": t_delay,
                        "Urgency": t_urgency,
                        "Task_Type_Exam": task_type_exam,
                        "Task_Type_Project": task_type_project,
                        "Task_Type_Quiz": task_type_quiz,
                        "Task_Type_Report": task_type_report
                    }
                    valid, msg = is_valid_against_csv(new_row)
                    if not valid:
                        st.error(f"CSV Validation Failed: {msg}")
                    else:
                        new_data = pd.DataFrame([{
                            "Task": t_name.strip(),
                            "Module": t_mod.strip().upper(),
                            "Deadline": t_date,
                            "Priority": t_prio,
                            "Status": "Not Started",
                            "Progress": 0,
                            "Year": t_year,
                            "Semester": t_sem,
                            "Week_Released": t_week_released,
                            "Week_Deadline": t_week_deadline,
                            "Current_Week": t_current_week,
                            "Weeks_Left": t_weeks_left,
                            "Weight": t_weight,
                            "Difficulty": t_diff,
                            "Estimated_Hours": t_hours,
                            "Current_Workload": t_workload,
                            "Procrastination_Score": t_procrast,
                            "Avg_Delay_History": t_delay,
                            "Urgency": t_urgency,
                            "Task_Type_Exam": task_type_exam,
                            "Task_Type_Project": task_type_project,
                            "Task_Type_Quiz": task_type_quiz,
                            "Task_Type_Report": task_type_report
                        }])
                        st.session_state.tasks_df = pd.concat([st.session_state.tasks_df, new_data], ignore_index=True)
                        st.success("Task added successfully!")
                        st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_table:
        st.markdown("### 📋 Active Workload")

        # Style the dataframe editor
        edited_df = st.data_editor(
            st.session_state.tasks_df,
            use_container_width=True,
            num_rows="dynamic",
            height=600,
            column_config={
                "Priority": st.column_config.SelectboxColumn(
                    "Priority",
                    options=["Critical", "High", "Medium", "Low"],
                    width="small",
                    required=True,
                ),
                "Status": st.column_config.SelectboxColumn(
                    "Status",
                    options=["Not Started", "In Progress", "Done"],
                    width="small",
                ),
                "Progress": st.column_config.ProgressColumn(
                    "Completion",
                    min_value=0,
                    max_value=100,
                    format="%f%%",
                ),
                "Deadline": st.column_config.DateColumn(
                    "Due Date",
                    format="DD MMM YYYY",
                )
            }
        )
        st.session_state.tasks_df = edited_df

# -----------------------------------------------------------------------------
# 10. MODULE: COURSE RECOMMENDER
# -----------------------------------------------------------------------------
elif menu == "Course Recommender":

    c1, c2 = st.columns([1, 2])

    with c1:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown("#### 🎓 Student Profile")
        gpa = st.slider("Current GPA", 0.0, 4.0, 3.2)
        interests = st.multiselect(
            "Academic Interests",
            ["Data Science", "Computer Vision", "Robotics", "Project Mgmt", "Cyber Security"],
            default=["Data Science"]
        )

        if st.button("Generate AI Recommendations", use_container_width=True):
            if not interests:
                st.warning("Please select at least one academic interest to generate recommendations.")
            else:
                st.session_state.show_recs = True

        st.markdown("---")
        if st.session_state.saved_courses:
            st.markdown("#### ✅ Saved Courses")
            for s in st.session_state.saved_courses:
                st.markdown(f"- {s}")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        if st.session_state.get("show_recs"):
            st.markdown("### Recommended Modules for Next Semester")

            # Mock Recommendations
            recs = [
                {"code": "CS410", "name": "Advanced Deep Learning", "match": 96, "tags": ["AI", "Hard"],
                 "reason": "Matches your interest in Computer Vision."},
                {"code": "DS302", "name": "Big Data Engineering", "match": 88, "tags": ["Data", "Medium"],
                 "reason": "Highly relevant to industry trends."},
                {"code": "MGMT201", "name": "Tech Leadership", "match": 75, "tags": ["Soft Skills", "Easy"],
                 "reason": "Recommended to balance workload risk."}
            ]

            for r in recs:
                match_color = "#00d2b4" if r['match'] >= 90 else "#3b82f6" if r['match'] >= 80 else "#f59e0b"
                st.markdown(f"""
                <div class="css-card" style="border-left: 3px solid {match_color};">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom: 8px;">
                        <div>
                            <div style="font-family:var(--mono); font-size:0.7rem; color:{match_color}; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:4px;">{r['code']}</div>
                            <div style="font-family:var(--font); font-weight:700; font-size:1.05rem; color:#f1f5f9;">{r['name']}</div>
                        </div>
                        <div style="
                            background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
                            padding: 6px 12px; border-radius: 8px; text-align:center; flex-shrink:0; margin-left:12px;
                        ">
                            <div style="font-family:var(--mono); font-weight:700; font-size:1.1rem; color:{match_color};">{r['match']}%</div>
                            <div style="font-size:0.6rem; text-transform:uppercase; letter-spacing:0.1em; color:#475569;">Match</div>
                        </div>
                    </div>
                    <p style="color: #64748b; font-size: 0.84rem; margin: 0 0 12px; line-height: 1.5;">{r['reason']}</p>
                    <div style="display:flex; gap:6px; flex-wrap:wrap;">
                        {''.join([f'<span style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); color:#94a3b8; padding:2px 10px; border-radius:6px; font-size:0.72rem; font-weight:600;">{t}</span>' for t in r['tags']])}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if st.button(f"Add {r['code']} to Plan", key=r['code']):
                    if r['code'] not in st.session_state.saved_courses:
                        st.session_state.saved_courses.append(r['code'])
                        st.success(f"Added {r['code']}")
                        st.rerun()
        else:
            st.info("👈 Select your interests and click 'Generate' to see AI suggestions.")