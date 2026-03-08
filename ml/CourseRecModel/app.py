import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from db_manager import StudentDB

# ── Import the separated PHI class ────────────────────────────────────────
from PHI_INT import GPAInterventionSystem

# ============================================================================
# PART 1: MODEL LOADING
# ============================================================================

db = StudentDB(db_path=r"D:\IIT Stuff\Group Project\DomainSpecifiedAIAssistant project\Project\domain-ai\ml\CourseRecModel\students.db")
MODEL_3YR_PATH = "3yrgpa_predictor_model.pkl"
MODEL_4YR_PATH = "4yrgpa_predictor_model.pkl"
DATA_PATH      = "final_dataset.csv"

if not os.path.exists(MODEL_3YR_PATH) or not os.path.exists(MODEL_4YR_PATH):
    st.error(f"Error: Required model files not found.")
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


# ============================================================================
# PART 2: FEATURE DEFINITIONS
# ============================================================================

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

grade_point_map = {'A': 4.0, 'B': 3.0, 'C': 2.0, 'D': 1.0, 'NOT_TAKEN': 0.0}
grade_point_options = sorted(list(grade_point_map.keys()),
                             key=lambda x: grade_point_map[x], reverse=True)


# ============================================================================
# PART 3: INPUT FUNCTION
# ============================================================================

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

    st.header(f"🎓 Course Grades (Previous Stages)")
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
        'Stage':                                   target_year,
        'hours_per_week':                          hours_per_week,
        'academic_stress':                         academic_stress,
        'struggle_with_managing':                  struggle_with_managing,
        'start_assignments_closer_deadline':       start_assignments_closer_deadline,
        'Skill_Programming':                       skill_programming,
        'Skill_Math':                              skill_math,
        'Skill_Technical_Comm':                    skill_tech_comm,
        'Skill_Web_Development':                   skill_web_dev,
        'Balancing multiple courses or projects':  1 if challenge_balance  else 0,
        'Not knowing how to prioritize task':      1 if challenge_priority else 0,
        'Starting tasks too late':                 1 if challenge_starting else 0,
        'Year1_GPA':                               year1_gpa,
        'Year2_GPA':                               year2_gpa,
        'Year3_GPA':                               year3_gpa,
        **grade_inputs
    }

    final_data = {key: user_data[key] for key in current_features}
    input_df   = pd.DataFrame([final_data], columns=current_features)
    return input_df, user_data


# ============================================================================
# PART 4: STREAMLIT UI
# ============================================================================

st.set_page_config(
    page_title="GPA Prediction with AI Advisor",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🎓"
)

st.title("🎓 Student GPA Predictor & Advisor")
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

# ── Instantiate PHI system (imported from phi_intervention.py) ─────────────
intervention_system = GPAInterventionSystem(current_model, full_data, target_year, db=db)

st.markdown("---")
input_df, user_data_dict = get_user_inputs(target_year, current_features)

if st.button("✨ Predict My GPA & Get Personalized Advice", type="primary", use_container_width=True):
    with st.spinner("🔮 Analysing your profile and generating recommendations..."):

        # ── Predict ───────────────────────────────────────────────────────
        predicted_gpa = intervention_system.predict_gpa(input_df)
        db.save_prediction({"target_year": target_year,
                            "predicted_gpa": predicted_gpa,
                            "user_data": user_data_dict})

        # ── Generate PHI recommendations ──────────────────────────────────
        current_gpa, phi_interventions = intervention_system.generate_interventions(
            user_data_dict, top_k=5
        )

        # ── SECTION 1: GPA Result ─────────────────────────────────────────
        st.markdown("## Your GPA Prediction")
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.metric(label=f"Predicted Year {target_year} GPA",
                      value=f"{predicted_gpa:.3f}")
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

        # ── SECTION 2: PHI Intervention Cards ────────────────────────────
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

        # ── SECTION 3: Raw Input Table ────────────────────────────────────
        st.markdown("### Input Data")
        st.dataframe(input_df.T, use_container_width=True)

        # ── SECTION 4: Export ─────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 💾 Export Your Results")
        col_e1, col_e2 = st.columns(2)

        with col_e1:
            summary_df = pd.DataFrame({
                "Predicted GPA":            [f"{predicted_gpa:.3f}"],
                "Risk Level":               ["On Track" if predicted_gpa >= intervention_system.gpa_threshold else "Needs Attention"],
                "Number of Recommendations":[len(phi_interventions)],
                "Top Intervention":         [phi_interventions[0]['name'] if phi_interventions else "None needed"],
                "Expected Improvement":     [f"+{phi_interventions[0]['gpa_improvement']:.3f}" if phi_interventions else "N/A"],
                "Analysis Date":            [datetime.now().strftime("%Y-%m-%d %H:%M")]
            })
            st.download_button("📄 Download Summary (CSV)",
                               summary_df.to_csv(index=False),
                               file_name=f"gpa_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                               mime="text/csv")

        with col_e2:
            if phi_interventions:
                interventions_df = pd.DataFrame({
                    "Priority":        [i + 1 for i in range(len(phi_interventions))],
                    "Intervention":    [i['name']            for i in phi_interventions],
                    "Domain":          [i['domain']           for i in phi_interventions],
                    "FPP Score":       [f"{i['fpp_score']:.2f}" for i in phi_interventions],
                    "GPA Improvement": [f"+{i['gpa_improvement']:.3f}" for i in phi_interventions],
                    "Description":     [i['description']      for i in phi_interventions]
                })
                st.download_button("📋 Download Full Plan (CSV)",
                                   interventions_df.to_csv(index=False),
                                   file_name=f"improvement_plan_{datetime.now().strftime('%Y%m%d')}.csv",
                                   mime="text/csv")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Powered by Random Forest ML (R² > 0.95) + Probabilistic Helpful Interventions (PHI) Framework"
    "</div>",
    unsafe_allow_html=True
)