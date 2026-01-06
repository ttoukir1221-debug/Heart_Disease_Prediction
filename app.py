import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# Custom CSS for Premium UI
# ==========================================
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }

    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #e94560 0%, #ff6b6b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 1rem;
    }

    .sub-header {
        text-align: center;
        color: #a0a0a0;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }

    /* Card styling */
    .stCard {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 2rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Risk indicator styling */
    .risk-low {
        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 10px 40px rgba(0, 184, 148, 0.3);
    }

    .risk-medium {
        background: linear-gradient(135deg, #fdcb6e 0%, #f39c12 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 10px 40px rgba(253, 203, 110, 0.3);
    }

    .risk-high {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 10px 40px rgba(231, 76, 60, 0.3);
    }

    /* Metric card */
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #e94560;
    }

    .metric-label {
        color: #a0a0a0;
        font-size: 1rem;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #e94560 0%, #ff6b6b 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 50px;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(233, 69, 96, 0.4);
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(22, 33, 62, 0.95);
    }

    /* Input styling */
    .stSelectbox, .stSlider, .stNumberInput {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
    }

    /* Section headers */
    .section-header {
        color: #e94560;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(233, 69, 96, 0.3);
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #666;
        padding: 2rem;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# Load Models
# ==========================================
@st.cache_resource
def load_models():
    soft_voting = joblib.load('model/soft_voting_model.pkl')
    stacking = joblib.load('model/stacking_model.pkl')
    feature_names = joblib.load('model/feature_names.pkl')
    return soft_voting, stacking, feature_names

try:
    soft_voting_model, stacking_model, feature_names = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"Error loading models: {e}")

# ==========================================
# Header
# ==========================================
st.markdown('<h1 class="main-header">❤️ Heart Disease Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced ML-powered health risk assessment using ensemble learning</p>', unsafe_allow_html=True)

# ==========================================
# Sidebar - Model Selection
# ==========================================
with st.sidebar:
    st.markdown("## 🔧 Settings")
    st.markdown("---")

    model_choice = st.selectbox(
        "🤖 Select Model",
        ["Stacking Ensemble (Recommended)", "Soft Voting"],
        help="Stacking has 78% recall (detects 6x more cases), Soft Voting has higher accuracy but only 7% recall"
    )

    st.markdown("---")
    st.markdown("### 📊 Model Performance")

    if model_choice == "Stacking Ensemble (Recommended)":
        st.metric("Recall (Detection)", "78%", delta="Best ⭐")
        st.metric("Accuracy", "87%")
        st.metric("ROC-AUC", "0.800")
        st.success("✅ Better for medical screening")
    else:
        st.metric("Recall (Detection)", "7%", delta="-34%", delta_color="inverse")
        st.metric("Accuracy", "91.1%")
        st.metric("ROC-AUC", "0.834")
        st.warning("⚠️ Misses 93% of heart disease cases")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info("This app uses machine learning to predict heart disease risk. Stacking is recommended as it detects 6x more heart disease cases.")

# ==========================================
# Main Form
# ==========================================
if models_loaded:

    # Create two columns for input
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">📋 Personal Information</p>', unsafe_allow_html=True)

        # Age Category
        age_category = st.selectbox(
            "🎂 Age Category",
            ["Young", "Adult", "Old", "Very Old"],
            index=1,
            help="Young: 18-39, Adult: 40-59, Old: 60-79, Very Old: 80+"
        )

        # Sex
        sex = st.selectbox(
            "👤 Sex",
            ["Male", "Female"],
            help="Biological sex"
        )

        # Race
        race = st.selectbox(
            "🌍 Race",
            ["White", "Black", "Asian", "Hispanic", "American Indian/Alaskan Native", "Other"]
        )

        # BMI
        bmi = st.slider(
            "⚖️ BMI (Body Mass Index)",
            min_value=10.0,
            max_value=60.0,
            value=25.0,
            step=0.1,
            help="Normal: 18.5-24.9, Overweight: 25-29.9, Obese: 30+"
        )

        st.markdown('<p class="section-header">🏃 Lifestyle</p>', unsafe_allow_html=True)

        # Smoking
        smoking = st.selectbox(
            "🚬 Smoking",
            ["No", "Yes"],
            help="Have you smoked at least 100 cigarettes in your life?"
        )

        # Alcohol
        alcohol = st.selectbox(
            "🍺 Alcohol Drinking",
            ["No", "Yes"],
            help="Heavy drinkers (adult men >14/week, women >7/week)"
        )

        # Physical Activity
        physical_activity = st.selectbox(
            "🏋️ Physical Activity",
            ["Yes", "No"],
            help="Any physical activity in past 30 days?"
        )

        # Sleep Time
        sleep_time = st.slider(
            "😴 Sleep Time (hours/day)",
            min_value=1,
            max_value=24,
            value=7,
            help="Average hours of sleep per day"
        )

    with col2:
        st.markdown('<p class="section-header">🏥 Health Conditions</p>', unsafe_allow_html=True)

        # General Health
        gen_health = st.selectbox(
            "💪 General Health",
            ["Excellent", "Very good", "Good", "Fair", "Poor"],
            index=2
        )

        # Diabetic
        diabetic = st.selectbox(
            "🩸 Diabetic Status",
            ["No", "Yes", "No, borderline diabetes", "Yes (during pregnancy)"]
        )

        # Stroke
        stroke = st.selectbox(
            "🧠 Ever Had Stroke?",
            ["No", "Yes"]
        )

        # Difficulty Walking
        diff_walking = st.selectbox(
            "🚶 Difficulty Walking?",
            ["No", "Yes"],
            help="Do you have serious difficulty walking or climbing stairs?"
        )

        st.markdown('<p class="section-header">📊 Health Metrics</p>', unsafe_allow_html=True)

        # Physical Health
        physical_health = st.slider(
            "🤕 Physical Health (bad days/month)",
            min_value=0,
            max_value=30,
            value=0,
            help="Days of poor physical health in past 30 days"
        )

        # Mental Health
        mental_health = st.slider(
            "🧘 Mental Health (bad days/month)",
            min_value=0,
            max_value=30,
            value=0,
            help="Days of poor mental health in past 30 days"
        )

        st.markdown('<p class="section-header">🩺 Other Conditions</p>', unsafe_allow_html=True)

        # Asthma
        asthma = st.selectbox("🫁 Asthma?", ["No", "Yes"])

        # Kidney Disease
        kidney_disease = st.selectbox("🫘 Kidney Disease?", ["No", "Yes"])

        # Skin Cancer
        skin_cancer = st.selectbox("☀️ Skin Cancer?", ["No", "Yes"])

    # ==========================================
    # Prediction Button
    # ==========================================
    st.markdown("<br>", unsafe_allow_html=True)

    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_button = st.button("🔮 Predict Heart Disease Risk", use_container_width=True)

    if predict_button:
        # ==========================================
        # Prepare Input Data
        # ==========================================

        # Binary encodings
        binary_map = {"No": 0, "Yes": 1}
        sex_map = {"Female": 0, "Male": 1}

        # Create input dictionary
        input_data = {
            'BMI': bmi,
            'Smoking': binary_map[smoking],
            'AlcoholDrinking': binary_map[alcohol],
            'Stroke': binary_map[stroke],
            'PhysicalHealth': physical_health,
            'MentalHealth': mental_health,
            'DiffWalking': binary_map[diff_walking],
            'Sex': sex_map[sex],
            'PhysicalActivity': binary_map[physical_activity],
            'SleepTime': sleep_time,
            'Asthma': binary_map[asthma],
            'KidneyDisease': binary_map[kidney_disease],
            'SkinCancer': binary_map[skin_cancer],

            # Age Category one-hot
            'AgeCategory_Adult': 1 if age_category == "Adult" else 0,
            'AgeCategory_Old': 1 if age_category == "Old" else 0,
            'AgeCategory_Very Old': 1 if age_category == "Very Old" else 0,
            'AgeCategory_Young': 1 if age_category == "Young" else 0,

            # Race one-hot
            'Race_American Indian/Alaskan Native': 1 if race == "American Indian/Alaskan Native" else 0,
            'Race_Asian': 1 if race == "Asian" else 0,
            'Race_Black': 1 if race == "Black" else 0,
            'Race_Hispanic': 1 if race == "Hispanic" else 0,
            'Race_Other': 1 if race == "Other" else 0,
            'Race_White': 1 if race == "White" else 0,

            # Diabetic one-hot
            'Diabetic_No': 1 if diabetic == "No" else 0,
            'Diabetic_No, borderline diabetes': 1 if diabetic == "No, borderline diabetes" else 0,
            'Diabetic_Yes': 1 if diabetic == "Yes" else 0,
            'Diabetic_Yes (during pregnancy)': 1 if diabetic == "Yes (during pregnancy)" else 0,

            # General Health one-hot
            'GenHealth_Excellent': 1 if gen_health == "Excellent" else 0,
            'GenHealth_Fair': 1 if gen_health == "Fair" else 0,
            'GenHealth_Good': 1 if gen_health == "Good" else 0,
            'GenHealth_Poor': 1 if gen_health == "Poor" else 0,
            'GenHealth_Very good': 1 if gen_health == "Very good" else 0,
        }

        # Create DataFrame with correct column order
        input_df = pd.DataFrame([input_data])[feature_names]

        # ==========================================
        # Make Prediction
        # ==========================================
        if model_choice == "Stacking Ensemble (Recommended)":
            model = stacking_model
        else:
            model = soft_voting_model

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        risk_percentage = probability * 100

        # ==========================================
        # Display Results
        # ==========================================
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("## 📊 Prediction Results")

        # Result columns
        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            if risk_percentage < 30:
                st.markdown(f'''
                <div class="risk-low">
                    <div style="font-size: 3rem;">✅</div>
                    <div style="font-size: 2rem; margin: 1rem 0;">LOW RISK</div>
                    <div style="font-size: 1.2rem;">You appear to be healthy!</div>
                </div>
                ''', unsafe_allow_html=True)
            elif risk_percentage < 60:
                st.markdown(f'''
                <div class="risk-medium">
                    <div style="font-size: 3rem;">⚠️</div>
                    <div style="font-size: 2rem; margin: 1rem 0;">MODERATE RISK</div>
                    <div style="font-size: 1.2rem;">Consider lifestyle changes</div>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="risk-high">
                    <div style="font-size: 3rem;">🚨</div>
                    <div style="font-size: 2rem; margin: 1rem 0;">HIGH RISK</div>
                    <div style="font-size: 1.2rem;">Please consult a doctor</div>
                </div>
                ''', unsafe_allow_html=True)

        with res_col2:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-label">Risk Probability</div>
                <div class="metric-value">{risk_percentage:.1f}%</div>
                <div class="metric-label">of heart disease</div>
            </div>
            ''', unsafe_allow_html=True)

        with res_col3:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-label">Model Used</div>
                <div style="font-size: 1.5rem; color: #e94560; font-weight: bold; margin: 0.5rem 0;">
                    {"Stacking" if model_choice == "Stacking Ensemble (Recommended)" else "Soft Voting"}
                </div>
                <div class="metric-label">Ensemble Classifier</div>
            </div>
            ''', unsafe_allow_html=True)

        # Progress bar visualization
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Risk Level Indicator")

        # Custom progress bar
        if risk_percentage < 30:
            bar_color = "#00b894"
        elif risk_percentage < 60:
            bar_color = "#fdcb6e"
        else:
            bar_color = "#e74c3c"

        st.markdown(f'''
        <div style="background: rgba(255,255,255,0.1); border-radius: 10px; height: 30px; overflow: hidden;">
            <div style="background: {bar_color}; width: {risk_percentage}%; height: 100%;
                        border-radius: 10px; transition: width 0.5s ease;
                        display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
                {risk_percentage:.1f}%
            </div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 5px; color: #888;">
            <span>0% (Low Risk)</span>
            <span>50% (Moderate)</span>
            <span>100% (High Risk)</span>
        </div>
        ''', unsafe_allow_html=True)

        # Disclaimer
        st.markdown("<br>", unsafe_allow_html=True)
        st.warning("⚠️ **Disclaimer:** This prediction is for informational purposes only and should not replace professional medical advice. Please consult a healthcare provider for accurate diagnosis.")

else:
    st.error("❌ Models could not be loaded. Please ensure model files exist in the 'model/' directory.")

# ==========================================
# Footer
# ==========================================
st.markdown('''
<div class="footer">
    <p>Built with ❤️ using Streamlit & Scikit-learn</p>
    <p style="font-size: 0.8rem;">Heart Disease Prediction System v1.0</p>
</div>
''', unsafe_allow_html=True)
