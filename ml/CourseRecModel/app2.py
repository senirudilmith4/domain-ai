import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="Course Recommender AI", page_icon="🎓", layout="centered")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { 
        width: 100%; border-radius: 5px; height: 3em; 
        background-color: #007bff; color: white; font-weight: bold;
    }
    .result-box { 
        padding: 20px; border-radius: 10px; background-color: white; 
        border-left: 5px solid #28a745; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); 
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODELS ---
@st.cache_resource
def load_assets():
    try:
        # Load both KNN models
        knn3 = joblib.load('Stage3_elective_recommender_knn.pkl')
        knn4 = joblib.load('Stage4_elective_recommender_knn.pkl')
        
        # Load the separate scalers (ensure you have saved a specific scaler for Stage 4)
        scaler3 = joblib.load('gpa_scaler.pkl') # Assuming this is for Stage 3
        scaler4 = joblib.load('gpa_scaler4.pkl') # You need a scaler trained on 4 features
        
        dataset = pd.read_csv('final_dataset.csv')
        return knn3, knn4, scaler3, scaler4, dataset
    except Exception as e:
        st.error(f"⚠️ Error loading files: {e}")
        return None, None, None, None, None

knn3, knn4, scaler3, scaler4, df = load_assets()

# --- HEADER ---
st.title("🎓 Course Recommender System")
st.write("Personalized elective suggestions for Stage 3 and Stage 4 students.")

if knn3 is not None and knn4 is not None:
    # --- INPUT SECTION ---
    st.subheader("Student Academic Profile")
    
    selected_stage = st.selectbox("Current Academic Stage", options=[3, 4], index=0)
    
    # Dynamic GPA Inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        y1_gpa = st.number_input("Year 1 GPA", 0.0, 4.0, 3.0, 0.01)
    with col2:
        y2_gpa = st.number_input("Year 2 GPA", 0.0, 4.0, 3.0, 0.01)
    
    # Year 3 GPA only shows if Stage 4 is selected
    y3_gpa = 0.0
    if selected_stage == 4:
        with col3:
            y3_gpa = st.number_input("Year 3 GPA", 0.0, 4.0, 3.0, 0.01)

    # --- RECOMMENDATION LOGIC ---
    if st.button("Generate Recommendation"):
        with st.spinner('Analyzing peer patterns...'):
            try:
                if selected_stage == 3:
                    # Stage 3 Logic: Uses [Stage, Y1, Y2]
                    X_input = np.array([[3, y1_gpa, y2_gpa]])
                    X_scaled = scaler3.transform(X_input)
                    distances, indices = knn3.kneighbors(X_scaled)
                    
                    neighbors = df.iloc[indices[0]]
                    avg_3602 = neighbors[neighbors['Grade_CM3602'] > 0]['Grade_CM3602'].mean()
                    avg_3603 = neighbors[neighbors['Grade_CM3603'] > 0]['Grade_CM3603'].mean()
                    
                    if (avg_3602 or 0) > (avg_3603 or 0):
                        rec_course, score = "CM3602: Internet of Things (IoT)", avg_3602
                    else:
                        rec_course, score = "CM3603: Edge Artificial Intelligence", avg_3603

                else:
                    # Stage 4 Logic: Uses [Stage, Y1, Y2, Y3]
                    X_input = np.array([[4, y1_gpa, y2_gpa, y3_gpa]])
                    X_scaled = scaler4.transform(X_input)
                    distances, indices = knn4.kneighbors(X_scaled)
                    
                    neighbors = df.iloc[indices[0]]
                    avg_4606 = neighbors[neighbors['Grade_CM4606'] > 0]['Grade_CM4606'].mean()
                    avg_4603 = neighbors[neighbors['Grade_CM4603'] > 0]['Grade_CM4603'].mean()
                    
                    if (avg_4606 or 0) > (avg_4603 or 0):
                        rec_course, score = "CM4606: Machine Vision", avg_4606
                    else:
                        rec_course, score = "CM4603: Natural Language Processing", avg_4603

                # UI Result Display
                st.markdown(f"""
                <div class="result-box">
                    <h3 style='color: #28a745;'>Top Pick: {rec_course}</h3>
                    <p>Students with a similar GPA profile to yours performed best in this elective.</p>
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