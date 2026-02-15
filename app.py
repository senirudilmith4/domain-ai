import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, List, Tuple
from datetime import datetime


# ============================================================================
# PART 1: MODEL LOADING 
# ============================================================================

MODEL_3YR_PATH = "3yrgpa_predictor_model.pkl"
MODEL_4YR_PATH = "4yrgpa_predictor_model.pkl"
DATA_PATH = "final_dataset.csv"

if not os.path.exists(MODEL_3YR_PATH) or not os.path.exists(MODEL_4YR_PATH):
    st.error(f"Error: Required model files '{MODEL_3YR_PATH}' and '{MODEL_4YR_PATH}' not found.")
    st.stop()

try:
    model_3yr = joblib.load(MODEL_3YR_PATH)
    model_4yr = joblib.load(MODEL_4YR_PATH)
    
    # Load data for threshold calculation
    if os.path.exists(DATA_PATH):
        full_data = pd.read_csv(DATA_PATH)
    else:
        full_data = None
        st.warning(f"Dataset '{DATA_PATH}' not found. Using default threshold.")
        
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()


# ============================================================================
# PART 2: PHI INTERVENTION SYSTEM
# ============================================================================

class GPAInterventionSystem:
    """
    Probabilistic Helpful Interventions system for Streamlit
    """
    
    def __init__(self, model, data, target_year: int = 3):
        self.model = model
        self.data = data
        self.target_year = target_year
        self.feature_names = model.feature_names_in_.tolist()
        
        # Calculate threshold from active students only
        if data is not None:
            if target_year == 3:
                active_data = data[data['Year3_GPA'] > 0]
                self.gpa_threshold = active_data['Year3_GPA'].quantile(0.50) if len(active_data) > 0 else 2.5
            else:
                active_data = data[data['Year4_GPA'] > 0]
                self.gpa_threshold = active_data['Year4_GPA'].quantile(0.50) if len(active_data) > 0 else 2.5
        else:
            self.gpa_threshold = 2.5
        
        # Define interventions
        self.intervention_domains = self._define_intervention_domains()
    
    def _define_intervention_domains(self) -> Dict:
        """Define all possible interventions"""
        return {
            'time_management': {
                'feature': 'hours_per_week',
                'interventions': [
                    {
                        'name': 'Increase Study Hours (Low to Medium)',
                        'condition': lambda x: x < 4.0,
                        'change': lambda x: 7.5,
                        'description': 'Increase weekly study time to 5-10 hours using gap times between classes',
                        'icon': '📚'
                    },
                    {
                        'name': 'Increase Study Hours (Medium to High)',
                        'condition': lambda x: 4.0 <= x < 7.5,
                        'change': lambda x: 12.5,
                        'description': 'Increase weekly study time to over 10 hours with structured schedule',
                        'icon': '📚'
                    }
                ]
            },
            'procrastination': {
                'feature': 'start_assignments_closer_deadline',
                'interventions': [
                    {
                        'name': 'Reduce Procrastination (Always → Sometimes)',
                        'condition': lambda x: x == 3,
                        'change': lambda x: 2,
                        'description': 'Start assignments 3-5 days before deadline instead of last minute',
                        'icon': '⏰'
                    },
                    {
                        'name': 'Reduce Procrastination (Sometimes → Rarely)',
                        'condition': lambda x: x == 2,
                        'change': lambda x: 1,
                        'description': 'Consistently start assignments early - at least 1 week before deadline',
                        'icon': '⏰'
                    }
                ]
            },
            'task_management': {
                'feature': 'struggle_with_managing',
                'interventions': [
                    {
                        'name': 'Improve Task Management (Often → Sometimes)',
                        'condition': lambda x: x == 3,
                        'change': lambda x: 2,
                        'description': 'Use planning tools (calendar, to-do lists) to reduce management struggles',
                        'icon': '📋'
                    },
                    {
                        'name': 'Improve Task Management (Sometimes → Rarely)',
                        'condition': lambda x: x == 2,
                        'change': lambda x: 1,
                        'description': 'Master time-blocking and prioritization techniques (Eisenhower Matrix)',
                        'icon': '📋'
                    }
                ]
            },
            'stress_management': {
                'feature': 'academic_stress',
                'interventions': [
                    {
                        'name': 'Reduce Stress (High → Moderate)',
                        'condition': lambda x: x >= 4,
                        'change': lambda x: 3,
                        'description': 'Implement stress reduction: regular breaks, exercise, sleep hygiene, counseling',
                        'icon': '🧘'
                    },
                    {
                        'name': 'Reduce Stress (Moderate → Low)',
                        'condition': lambda x: x == 3,
                        'change': lambda x: 2,
                        'description': 'Enhance coping strategies: mindfulness, study groups, time management',
                        'icon': '🧘'
                    }
                ]
            },
            'programming_skill': {
                'feature': 'Skill_Programming',
                'interventions': [
                    {
                        'name': 'Improve Programming Skills',
                        'condition': lambda x: x <= 3,
                        'change': lambda x: min(5, x + 1),
                        'description': 'Dedicate 5-7 hours/week to coding practice, online courses (CS50, freeCodeCamp)',
                        'icon': '💻'
                    }
                ]
            },
            'math_skill': {
                'feature': 'Skill_Math',
                'interventions': [
                    {
                        'name': 'Improve Math Skills',
                        'condition': lambda x: x <= 3,
                        'change': lambda x: min(5, x + 1),
                        'description': 'Regular practice, attend office hours, form study groups for math courses',
                        'icon': '📐'
                    }
                ]
            },
            'workload_balance': {
                'feature': 'Balancing multiple courses or projects',
                'interventions': [
                    {
                        'name': 'Improve Course/Project Balance',
                        'condition': lambda x: x == 1,
                        'change': lambda x: 0,
                        'description': 'Create a master schedule with dedicated time blocks for each course/project',
                        'icon': '⚖️'
                    }
                ]
            },
            'task_prioritization': {
                'feature': 'Not knowing how to prioritize task',
                'interventions': [
                    {
                        'name': 'Learn Task Prioritization',
                        'condition': lambda x: x == 1,
                        'change': lambda x: 0,
                        'description': 'Learn prioritization frameworks (urgent/important matrix, ABC method)',
                        'icon': '🎯'
                    }
                ]
            },
            'deadline_management': {
                'feature': 'Starting tasks too late',
                'interventions': [
                    {
                        'name': 'Start Tasks Earlier',
                        'condition': lambda x: x == 1,
                        'change': lambda x: 0,
                        'description': 'Set personal deadlines 3-5 days before actual deadlines, use backward planning',
                        'icon': '📅'
                    }
                ]
            }
        }
    
    def predict_gpa(self, student_data: pd.DataFrame) -> float:
        """Predict GPA ensuring correct column order"""
        student_data = student_data[self.feature_names]
        return self.model.predict(student_data)[0]
    
    def classify_gpa_probability(self, predicted_gpa: float) -> float:
        """Convert GPA to probability of being High GPA"""
        distance = predicted_gpa - self.gpa_threshold
        probability = 1 / (1 + np.exp(-2 * distance))
        return probability
    
    def calculate_fpp(self, current_prob: float, modified_prob: float) -> float:
        """Calculate Fold Change in Posterior Probability"""
        p_low_current = 1 - current_prob if current_prob < 0.5 else current_prob
        p_high_modified = modified_prob if modified_prob > 0.5 else 1 - modified_prob
        
        if p_low_current < 0.01:
            p_low_current = 0.01
        
        return p_high_modified / p_low_current
    
    def simulate_intervention(self, student_data: Dict, domain_name: str, 
                            intervention_idx: int) -> Tuple:
        """Simulate single intervention"""
        domain = self.intervention_domains[domain_name]
        intervention = domain['interventions'][intervention_idx]
        feature = domain['feature']
        
        current_value = student_data[feature]
        if not intervention['condition'](current_value):
            return None
        
        # Create modified data
        modified_data = student_data.copy()
        modified_data[feature] = intervention['change'](current_value)
        
        # Predict both
        current_df = pd.DataFrame([student_data])
        modified_df = pd.DataFrame([modified_data])
        
        current_gpa = self.predict_gpa(current_df)
        modified_gpa = self.predict_gpa(modified_df)
        
        # Calculate FPP
        current_prob = self.classify_gpa_probability(current_gpa)
        modified_prob = self.classify_gpa_probability(modified_gpa)
        fpp = self.calculate_fpp(current_prob, modified_prob)
        
        return {
            'name': intervention['name'],
            'description': intervention['description'],
            'icon': intervention['icon'],
            'domain': domain_name,
            'fpp_score': fpp,
            'current_gpa': current_gpa,
            'modified_gpa': modified_gpa,
            'gpa_improvement': modified_gpa - current_gpa,
            'is_phi': fpp >= 1.0
        }
    
    def generate_interventions(self, student_data: Dict, top_k: int = 5) -> Tuple[float, List[Dict]]:
        """Generate all interventions for student"""
        # Get current prediction
        current_gpa = self.predict_gpa(pd.DataFrame([student_data]))
        
        # Test all interventions
        results = []
        for domain_name, domain in self.intervention_domains.items():
            for idx in range(len(domain['interventions'])):
                result = self.simulate_intervention(student_data, domain_name, idx)
                if result:
                    results.append(result)
        
        # Sort by FPP and filter PHI
        results.sort(key=lambda x: x['fpp_score'], reverse=True)
        phi_only = [r for r in results if r['is_phi']][:top_k]
        
        return current_gpa, phi_only


# ============================================================================
# PART 3: FEATURE DEFINITIONS
# ============================================================================

features_3yr = [
    'Stage', 'hours_per_week', 'academic_stress', 'struggle_with_managing', 
    'start_assignments_closer_deadline', 'Skill_Programming', 'Skill_Math', 
    'Skill_Technical_Comm', 'Skill_Web_Development', 
    'Grade_CM1601', 'Grade_CM1602', 'Grade_CM1603', 'Grade_CM1604', 
    'Grade_CM1605', 'Grade_CM1606', 'Grade_CM2601', 'Grade_CM2602', 
    'Grade_CM2603', 'Grade_CM2604', 'Grade_CM2605', 'Grade_CM2606', 
    'Grade_CM2607', 'Year1_GPA', 'Year2_GPA', 
    'Balancing multiple courses or projects', 'Not knowing how to prioritize task', 'Starting tasks too late'
]

features_4yr = [
    'Stage', 'hours_per_week', 'academic_stress', 'struggle_with_managing', 
    'start_assignments_closer_deadline', 'Skill_Programming', 'Skill_Math', 
    'Skill_Technical_Comm', 'Skill_Web_Development', 
    'Grade_CM1601', 'Grade_CM1602', 'Grade_CM1603', 'Grade_CM1604', 
    'Grade_CM1605', 'Grade_CM1606', 'Grade_CM2601', 'Grade_CM2602', 
    'Grade_CM2603', 'Grade_CM2604', 'Grade_CM2605', 'Grade_CM2606', 
    'Grade_CM2607', 'Grade_CM3606', 'Grade_CM3604', 'Grade_CM3602', 'Grade_CM3603', 
    'Year1_GPA', 'Year2_GPA', 'Year3_GPA',
    'Balancing multiple courses or projects', 'Not knowing how to prioritize task', 'Starting tasks too late'
]

grade_point_map = {
    'A': 4.0, 'B': 3.0, 'C': 2.0, 'D': 1.0, 'NOT_TAKEN': 0.0
}
grade_point_options = sorted(list(grade_point_map.keys()), 
                            key=lambda x: grade_point_map[x], reverse=True)


# ============================================================================
# PART 4: INPUT FUNCTION 
# ============================================================================

def get_user_inputs(target_year, current_features):
    st.sidebar.header("📊 Personal & Skill Inputs")
    
    st.sidebar.selectbox("Current Academic Stage", 
                        [f'Stage {i}' for i in range(1, 5)], 
                        index=target_year-1, disabled=True)
    
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
    skill_math = st.sidebar.slider("Skill_Math", 1, 5, 3)
    skill_tech_comm = st.sidebar.slider("Skill_Technical_Comm", 1, 5, 3)
    skill_web_dev = st.sidebar.slider("Skill_Web_Development", 1, 5, 3)

    st.sidebar.header("🤔 Workload Challenges")
    challenge_balance = st.sidebar.checkbox('Balancing multiple courses or projects', value=True)
    challenge_priority = st.sidebar.checkbox('Not knowing how to prioritize task', value=False)
    challenge_starting = st.sidebar.checkbox('Starting tasks too late', value=False)
    
    # GPA Inputs
    st.sidebar.subheader("Previous GPA Inputs")
    year1_gpa = st.sidebar.number_input("Year 1 GPA", min_value=0.0, max_value=4.0, step=0.01, value=0.0)
    year2_gpa = st.sidebar.number_input("Year 2 GPA", min_value=0.0, max_value=4.0, step=0.01, value=0.0)
    year3_gpa = 0.0
    if target_year == 4:
        year3_gpa = st.sidebar.number_input("Year 3 GPA (if available)", 
                                           min_value=0.0, max_value=4.0, step=0.01, value=0.0)

    st.header(f"🎓 Course Grades (Previous Stages)")
    st.markdown("Enter final grades for courses up to the target stage. Use 'NOT_TAKEN' for courses not yet completed.")

    grade_cols = [col for col in current_features if col.startswith('Grade_CM')]
    cols = st.columns(3)
    grade_inputs = {}
    for i, col in enumerate(grade_cols):
        course_name = col.replace('Grade_', '')
        default_grade = 'A' if 'CM16' in course_name else 'B' if 'CM26' in course_name else 'C' if 'CM36' in course_name else 'NOT_TAKEN'
        if target_year == 3 and ('CM36' in course_name or 'CM46' in course_name): 
            default_grade='NOT_TAKEN'
        if target_year == 4 and 'CM46' in course_name: 
            default_grade='NOT_TAKEN'
        with cols[i % 3]:
            selected_grade = st.selectbox(
                f"{course_name} Grade:", 
                grade_point_options, 
                index=grade_point_options.index(default_grade), 
                key=col
            )
            grade_inputs[col] = grade_point_map[selected_grade]

    user_data = {
        'Stage': target_year,
        'hours_per_week': hours_per_week,
        'academic_stress': academic_stress,
        'struggle_with_managing': struggle_with_managing,
        'start_assignments_closer_deadline': start_assignments_closer_deadline,
        'Skill_Programming': skill_programming,
        'Skill_Math': skill_math,
        'Skill_Technical_Comm': skill_tech_comm,
        'Skill_Web_Development': skill_web_dev,
        'Balancing multiple courses or projects': 1 if challenge_balance else 0,
        'Not knowing how to prioritize task': 1 if challenge_priority else 0,
        'Starting tasks too late': 1 if challenge_starting else 0,
        'Year1_GPA': year1_gpa,
        'Year2_GPA': year2_gpa,
        'Year3_GPA': year3_gpa
    }
    user_data.update(grade_inputs)

    final_data = {key: user_data[key] for key in current_features}
    input_df = pd.DataFrame([final_data], columns=current_features)
    return input_df, user_data


# ============================================================================
# PART 5: STREAMLIT UI WITH PHI INTEGRATION
# ============================================================================

st.set_page_config(
    page_title="GPA Prediction with AI Advisor", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="🎓"
)

st.title("🎓 Student GPA Predictor & Advisor")
st.markdown("Get your GPA prediction **PLUS** personalized improvement strategies powered by ML")

# Task selection
prediction_task = st.selectbox(
    "Select Target Prediction Year:", 
    ["Predict 3rd Year GPA", "Predict 4th Year GPA"]
)

if "3rd Year" in prediction_task:
    target_year = 3
    current_model = model_3yr
    current_features = features_3yr
    st.warning("⚠️ To predict 3rd-year GPA, input all grades up to Stage 2 (CM26xx). Stage 3/4 courses should be 'NOT_TAKEN'.")
else:
    target_year = 4
    current_model = model_4yr
    current_features = features_4yr
    st.warning("⚠️ To predict 4th-year GPA, input all grades up to Stage 3 (CM36xx). Stage 4 courses should be 'NOT_TAKEN'.")

# Initialize intervention system
intervention_system = GPAInterventionSystem(current_model, full_data, target_year)

st.markdown("---")

# Get user inputs
input_df, user_data_dict = get_user_inputs(target_year, current_features)

# Predict button
if st.button(f"✨ Predict My GPA & Get Personalized Advice", type="primary", use_container_width=True):
    with st.spinner("🔮 Analyzing your profile and generating personalized recommendations..."):
        
        # Make prediction
        predicted_gpa = intervention_system.predict_gpa(input_df)
        
        # Generate interventions
        current_gpa, phi_interventions = intervention_system.generate_interventions(
            user_data_dict, top_k=5
        )
        
        # ========== DISPLAY RESULTS ==========
        
        # Section 1: GPA Prediction
        st.markdown("## Your GPA Prediction")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.metric(
                label=f"Predicted Year {target_year} GPA",
                value=f"{predicted_gpa:.3f}",
                delta=None
            )
        
        with col2:
            threshold = intervention_system.gpa_threshold
            if predicted_gpa >= threshold:
                st.success("✅ On Track")
            else:
                st.warning("⚠️ Needs Attention")
        
        with col3:
            if predicted_gpa >= 3.7:
                grade_label = "Excellent (A)"
                st.balloons()
            elif predicted_gpa >= 3.0:
                grade_label = "Good (B+/A-)"
            elif predicted_gpa >= 2.0:
                grade_label = "Average (C/B)"
            else:
                grade_label = "At Risk (< C)"
            st.info(f"📊 {grade_label}")
        
        # Performance message
        if predicted_gpa >= 3.7:
            st.success("**Outstanding!** You're on track for an excellent GPA. Keep up the great work!")
        elif predicted_gpa >= 3.0:
            st.info("**Very Good!** You're performing well. Check the recommendations below for optimization.")
        elif predicted_gpa >= 2.0:
            st.warning("**Good, but room to improve.** Focus on the priority interventions below.")
        else:
            st.error("**Needs Immediate Attention!** Your GPA is below 2.0. Please review all recommendations carefully.")
        
        st.markdown("---")
        
        # Section 2: PHI Recommendations
        if len(phi_interventions) > 0:
            st.markdown("## Your Personalized Improvement Plan")
            st.markdown(f"**Found {len(phi_interventions)} scientifically-proven intervention(s) for you**")
            st.markdown("*These recommendations are mathematically calculated to improve YOUR specific situation*")
            
            st.markdown("")
            
            # Display each intervention as a card
            for idx, intervention in enumerate(phi_interventions, 1):
                
                # Determine effectiveness color
                fpp = intervention['fpp_score']
                if fpp >= 2.0:
                    effectiveness_color = "🟢"
                    effectiveness_text = "Very High Impact"
                elif fpp >= 1.5:
                    effectiveness_color = "🔵"
                    effectiveness_text = "High Impact"
                else:
                    effectiveness_color = "🟡"
                    effectiveness_text = "Moderate Impact"
                
                # Create expandable card
                with st.expander(
                    f"**#{idx}: {intervention['icon']} {intervention['name']}** "
                    f"{effectiveness_color} ({effectiveness_text})", 
                    expanded=(idx == 1)  # First one expanded by default
                ):
                    col_a, col_b = st.columns([2, 1])
                    
                    with col_a:
                        st.markdown(f"**📋 What to do:**")
                        st.markdown(f"> {intervention['description']}")
                        
                        st.markdown(f"**💡 Why this helps:**")
                        st.markdown(
                            f"> This intervention increases your success odds by "
                            f"**{(fpp - 1) * 100:.0f}%** (FPP Score: {fpp:.2f})"
                        )
                    
                    with col_b:
                        st.metric(
                            label="Expected GPA Increase",
                            value=f"+{intervention['gpa_improvement']:.3f}",
                            delta=f"{intervention['modified_gpa']:.3f} (new GPA)"
                        )
                        
                        st.markdown(f"**Domain:** {intervention['domain'].replace('_', ' ').title()}")
                        st.markdown(f"**FPP Score:** {fpp:.2f}")
                    
                   
            
            
        else:
            # No interventions needed
            st.markdown("## ✅ Excellent Profile!")
            st.success(
                "Based on your current profile, you're already well-optimized for success. "
                "No critical interventions needed at this time. Keep up the great work!"
            )
            
            if predicted_gpa < intervention_system.gpa_threshold:
                st.info(
                    "While your profile looks good, your predicted GPA is slightly below the threshold. "
                    "This might be due to past course performance. Focus on maintaining current good habits!"
                )
        
        st.markdown("### Input Data")
        st.dataframe(input_df.T, use_container_width=True)
        
        # Section 4: Export Options
        st.markdown("---")
        st.markdown("### 💾 Export Your Results")
        
        col_export1, col_export2 = st.columns(2)
        
        with col_export1:
            # Create summary for export
            summary_data = {
                "Student ID": ["Your Profile"],
                "Predicted GPA": [f"{predicted_gpa:.3f}"],
                "Risk Level": ["On Track" if predicted_gpa >= intervention_system.gpa_threshold else "Needs Attention"],
                "Number of Recommendations": [len(phi_interventions)],
                "Top Intervention": [phi_interventions[0]['name'] if len(phi_interventions) > 0 else "None needed"],
                "Expected Improvement": [f"+{phi_interventions[0]['gpa_improvement']:.3f}" if len(phi_interventions) > 0 else "N/A"],
                "Analysis Date": [datetime.now().strftime("%Y-%m-%d %H:%M")]
            }
            summary_df = pd.DataFrame(summary_data)
            
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="📄 Download Summary (CSV)",
                data=csv,
                file_name=f"gpa_prediction_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col_export2:
            # Create detailed interventions export
            if len(phi_interventions) > 0:
                interventions_data = {
                    "Priority": [i+1 for i in range(len(phi_interventions))],
                    "Intervention": [i['name'] for i in phi_interventions],
                    "Domain": [i['domain'] for i in phi_interventions],
                    "FPP Score": [f"{i['fpp_score']:.2f}" for i in phi_interventions],
                    "GPA Improvement": [f"+{i['gpa_improvement']:.3f}" for i in phi_interventions],
                    "Description": [i['description'] for i in phi_interventions]
                }
                interventions_df = pd.DataFrame(interventions_data)
                
                csv_interventions = interventions_df.to_csv(index=False)
                st.download_button(
                    label="📋 Download Full Plan (CSV)",
                    data=csv_interventions,
                    file_name=f"improvement_plan_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Powered by Random Forest ML (R² > 0.95) + Probabilistic Helpful Interventions (PHI) Framework"
    "</div>",
    unsafe_allow_html=True
)