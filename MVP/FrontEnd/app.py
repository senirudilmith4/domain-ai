import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
from datetime import date, datetime, timedelta

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
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

    /* Global Text Styles */
    .stApp {
        background-color: #0E1117;
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3 {
        color: #FAFAFA;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
    }

    /* Custom Card Styling (Dark Glassmorphism) */
    .css-card {
        background-color: #1a1c24;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .css-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 0 15px rgba(0, 173, 181, 0.2); /* Cyan Glow */
        border-color: #00ADB5;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #12141C;
        border-right: 1px solid #2d3748;
    }

    /* Metric Values */
    .metric-value {
        font-family: 'Roboto Mono', monospace;
        font-size: 28px;
        font-weight: bold;
        color: #00ADB5;
    }
    .metric-label {
        color: #A0AEC0;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Chat Messages */
    .stChatMessage {
        background-color: #1a1c24;
        border: 1px solid #2d3748;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #00ADB5;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
    }
    div.stButton > button:hover {
        background-color: #007981;
        color: white;
    }

    /* Input Fields */
    .stTextInput > div > div > input {
        background-color: #2d3748;
        color: white;
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


def ai_prioritize(df):
    """Sorts dataframe based on the proposal's predictive logic (Deadline + Importance)."""
    prio_map = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    df['p_score'] = df['Priority'].map(prio_map)
    # Sort by Deadline (asc) then Priority (Critical first)
    df = df.sort_values(by=['Deadline', 'p_score'])
    return df.drop(columns=['p_score'])


# -----------------------------------------------------------------------------
# 4. LOGIN SCREEN (Simulated)
# -----------------------------------------------------------------------------
if not st.session_state.login_state:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background-color: #1a1c24; padding: 40px; border-radius: 15px; border: 1px solid #2d3748; text-align: center;">
            <h2 style="color: #00ADB5;">AI Assistant Platform</h2>
            <p style="color: #A0AEC0; font-size: 0.9rem;">Informatics Institute of Technology<br>Robert Gordon University</p>
            <hr style="border-color: #2d3748;">
        </div>
        """, unsafe_allow_html=True)

        role = st.selectbox("Select User Role", ["Student", "Lecturer", "Admin Staff"])
        password = st.text_input("Password", type="password")

        if st.button("Authenticate", use_container_width=True):
            if password:
                st.session_state.login_state = True
                st.session_state.user_role = role
                st.rerun()
            else:
                st.warning("Enter any password to continue.")
    st.stop()

# -----------------------------------------------------------------------------
# 5. SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding-bottom: 20px;">
        <h3 style="margin-bottom: 5px;">Domain Specific AI</h3>
        <p style="color: #00ADB5; font-size: 12px; letter-spacing: 1px;">UNIVERSITY PLATFORM</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background: #2d3748; padding: 10px; border-radius: 8px; margin-bottom: 20px; text-align: center;">
        <span style="color: #A0AEC0; font-size: 12px;">CURRENT ROLE</span><br>
        <span style="color: white; font-weight: bold;">{st.session_state.user_role}</span>
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
st.markdown(f"## {menu}")
st.markdown(
    f"<p style='color: #718096; margin-top: -15px;'>Logged in as {st.session_state.user_role} | Academic Year 2025/26</p>",
    unsafe_allow_html=True)
st.markdown("---")

# -----------------------------------------------------------------------------
# 7. MODULE: DASHBOARD
# -----------------------------------------------------------------------------
if menu == "Dashboard":

    # Top Metrics Row
    m1, m2, m3 = st.columns(3)

    risk_val, risk_label = calculate_risk(st.session_state.tasks_df)
    next_due = st.session_state.tasks_df[st.session_state.tasks_df['Status'] != 'Done']['Deadline'].min()

    with m1:
        st.markdown(f"""
        <div class="css-card">
            <div class="metric-label">Workload Risk</div>
            <div class="metric-value" style="color: {'#FF5252' if risk_val > 80 else '#00ADB5'}">{risk_label}</div>
            <div style="font-size: 12px; color: #718096;">AI Prediction Model</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="css-card">
            <div class="metric-label">Next Deadline</div>
            <div class="metric-value" style="color: #FAFAFA">{next_due.strftime('%d %b')}</div>
            <div style="font-size: 12px; color: #718096;">{(next_due - date.today()).days} days remaining</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="css-card">
            <div class="metric-label">Pending Tasks</div>
            <div class="metric-value" style="color: #FAFAFA">{len(st.session_state.tasks_df[st.session_state.tasks_df['Status'] != 'Done'])}</div>
            <div style="font-size: 12px; color: #718096;"> Across all modules</div>
        </div>
        """, unsafe_allow_html=True)

    # Charts Row
    c1, c2 = st.columns([1, 2])

    with c1:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown("#### 📉 Stress & Workload Gauge")

        # Dark Theme Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_val,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [None, 100], 'tickcolor': "white"},
                'bar': {'color': "#00ADB5"},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "#2d3748",
                'steps': [
                    {'range': [0, 50], 'color': "#1a1c24"},
                    {'range': [50, 80], 'color': "#2d3748"},
                    {'range': [80, 100], 'color': "#4a5568"}],
            }))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white", 'family': "Inter"}, height=250,
                          margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown("#### 📊 Task Completion Velocity")

        # Mock Data for Chart
        chart_data = pd.DataFrame({
            "Day": ["Mon", "Tue", "Wed", "Thu", "Fri"],
            "Tasks Completed": [2, 1, 3, 0, 4],
            "New Tasks": [1, 2, 1, 5, 2]
        })

        fig2 = px.bar(chart_data, x="Day", y=["Tasks Completed", "New Tasks"],
                      barmode='group', color_discrete_sequence=["#00ADB5", "#718096"])

        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "#A0AEC0"},
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig2, use_container_width=True)
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
            t_name = st.text_input("Task Description")
            t_mod = st.text_input("Module Code")
            t_date = st.date_input("Due Date")
            t_prio = st.selectbox("Priority", ["Critical", "High", "Medium", "Low"])
            if st.form_submit_button("Add to Workflow"):
                new_data = pd.DataFrame([{
                    "Task": t_name, "Module": t_mod, "Deadline": t_date,
                    "Priority": t_prio, "Status": "Not Started", "Progress": 0
                }])
                st.session_state.tasks_df = pd.concat([st.session_state.tasks_df, new_data], ignore_index=True)
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
                # Custom HTML Card for Course
                st.markdown(f"""
                <div class="css-card" style="border-left: 4px solid #00ADB5;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h3 style="margin:0; color: #00ADB5;">{r['code']}</h3>
                        <span style="background:#2d3748; padding:5px 10px; border-radius:10px; font-weight:bold; color: #00ADB5;">{r['match']}% Match</span>
                    </div>
                    <h4 style="margin: 5px 0;">{r['name']}</h4>
                    <p style="color: #A0AEC0; font-size: 14px;">{r['reason']}</p>
                    <div style="margin-top:10px;">
                        {''.join([f'<span style="background:#4A5568; color:white; padding:2px 8px; border-radius:4px; font-size:12px; margin-right:5px;">{t}</span>' for t in r['tags']])}
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