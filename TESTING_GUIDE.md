# Heart Disease Prediction - Testing Guide

This guide provides test scenarios to help you understand how the prediction model works and verify its functionality.

---

## 🚀 How to Run the App

```bash
# Navigate to project folder
cd "e:\ML\Heart DIsease Prediction"

# Activate virtual environment
venv\Scripts\activate

# Run the app
streamlit run app.py
```

Then open your browser at: **http://localhost:8501**

---

## 📊 Understanding Risk Levels

| Risk Level        | Probability | Color     | Recommendation             |
| ----------------- | ----------- | --------- | -------------------------- |
| **Low Risk**      | 0% - 30%    | 🟢 Green  | You appear healthy         |
| **Moderate Risk** | 30% - 60%   | 🟡 Yellow | Consider lifestyle changes |
| **High Risk**     | 60% - 100%  | 🔴 Red    | Please consult a doctor    |

---

## ✅ Test Scenario 1: LOW RISK Patient

Use these inputs to see a **LOW RISK** prediction:

| Field                          | Value     |
| ------------------------------ | --------- |
| **Age Category**               | Young     |
| **Sex**                        | Female    |
| **Race**                       | White     |
| **BMI**                        | 22.0      |
| **Smoking**                    | No        |
| **Alcohol Drinking**           | No        |
| **Physical Activity**          | Yes       |
| **Sleep Time**                 | 7 hours   |
| **General Health**             | Excellent |
| **Diabetic Status**            | No        |
| **Stroke**                     | No        |
| **Difficulty Walking**         | No        |
| **Physical Health (bad days)** | 0         |
| **Mental Health (bad days)**   | 0         |
| **Asthma**                     | No        |
| **Kidney Disease**             | No        |
| **Skin Cancer**                | No        |

**Expected Result:** ~5-15% risk (LOW RISK - Green)

---

## ⚠️ Test Scenario 2: MODERATE RISK Patient

Use these inputs to see a **MODERATE RISK** prediction:

| Field                          | Value                   |
| ------------------------------ | ----------------------- |
| **Age Category**               | Adult                   |
| **Sex**                        | Male                    |
| **Race**                       | White                   |
| **BMI**                        | 28.0                    |
| **Smoking**                    | Yes                     |
| **Alcohol Drinking**           | No                      |
| **Physical Activity**          | No                      |
| **Sleep Time**                 | 6 hours                 |
| **General Health**             | Fair                    |
| **Diabetic Status**            | No, borderline diabetes |
| **Stroke**                     | No                      |
| **Difficulty Walking**         | No                      |
| **Physical Health (bad days)** | 10                      |
| **Mental Health (bad days)**   | 5                       |
| **Asthma**                     | No                      |
| **Kidney Disease**             | No                      |
| **Skin Cancer**                | No                      |

**Expected Result:** ~35-55% risk (MODERATE RISK - Yellow)

---

## 🚨 Test Scenario 3: HIGH RISK Patient

Use these inputs to see a **HIGH RISK** prediction:

| Field                          | Value    |
| ------------------------------ | -------- |
| **Age Category**               | Very Old |
| **Sex**                        | Male     |
| **Race**                       | White    |
| **BMI**                        | 35.0     |
| **Smoking**                    | Yes      |
| **Alcohol Drinking**           | Yes      |
| **Physical Activity**          | No       |
| **Sleep Time**                 | 4 hours  |
| **General Health**             | Poor     |
| **Diabetic Status**            | Yes      |
| **Stroke**                     | Yes      |
| **Difficulty Walking**         | Yes      |
| **Physical Health (bad days)** | 25       |
| **Mental Health (bad days)**   | 20       |
| **Asthma**                     | Yes      |
| **Kidney Disease**             | Yes      |
| **Skin Cancer**                | No       |

**Expected Result:** ~65-85% risk (HIGH RISK - Red)

---

## 🔬 Key Risk Factors

The model considers these as **high-impact risk factors**:

### High Risk Indicators (increase risk significantly)

- ⬆️ **Age**: Very Old (80+) > Old (60-79) > Adult (40-59)
- ⬆️ **Stroke History**: Previous stroke greatly increases risk
- ⬆️ **Diabetic**: Having diabetes increases risk
- ⬆️ **Poor General Health**: Self-reported poor health
- ⬆️ **Smoking**: History of smoking
- ⬆️ **High BMI**: Obesity (BMI > 30)
- ⬆️ **Difficulty Walking**: Mobility issues
- ⬆️ **Kidney Disease**: Pre-existing kidney problems

### Protective Factors (lower risk)

- ⬇️ **Young Age**: 18-39 years
- ⬇️ **Physical Activity**: Regular exercise
- ⬇️ **Good Sleep**: 7-8 hours per night
- ⬇️ **Excellent/Very Good Health**: Good overall health
- ⬇️ **Normal BMI**: 18.5-24.9
- ⬇️ **No Smoking**: Never smoked or quit

---

## 🧪 Quick Tests to Try

### Test 1: Age Impact

Keep everything at low-risk values but change only **Age Category**:

- Young → ~10% risk
- Adult → ~15% risk
- Old → ~25% risk
- Very Old → ~35% risk

### Test 2: Smoking Impact

Keep everything at low-risk values but change only **Smoking**:

- No → ~10% risk
- Yes → ~18% risk

### Test 3: Stroke Impact

Keep everything at low-risk values but change only **Stroke**:

- No → ~10% risk
- Yes → ~40% risk

### Test 4: Combined Risk Factors

Add multiple risk factors to see cumulative effect:

- 1 risk factor → ~15-25%
- 2-3 risk factors → ~30-50%
- 4+ risk factors → ~55-80%

---

## 🔄 Model Comparison Test

Try the same inputs with both models:

| Model           | Best For                           | Expected Behavior               |
| --------------- | ---------------------------------- | ------------------------------- |
| **Soft Voting** | Better at detecting positive cases | May show slightly higher risk % |
| **Stacking**    | Higher overall accuracy            | May show slightly lower risk %  |

Switch models in the sidebar and click "Predict" again to compare!

---

## 📝 Notes

1. **Results may vary** slightly based on the combination of factors
2. The model learned from ~300,000 patient records
3. **This is NOT medical advice** - always consult a healthcare professional
4. Risk factors interact with each other (e.g., smoking + diabetes = higher combined risk)

---

## 🐛 Troubleshooting

| Issue             | Solution                                                             |
| ----------------- | -------------------------------------------------------------------- |
| App won't start   | Make sure venv is activated                                          |
| Model not loading | Check `model/` folder has all 3 .pkl files                           |
| Slow prediction   | First prediction loads model into memory, subsequent ones are faster |
| Error messages    | Restart the app with `streamlit run app.py`                          |

---

## 📂 Project Structure

```
Heart Disease Prediction/
├── app.py                    # Streamlit UI application
├── requirements.txt          # Python dependencies
├── TESTING_GUIDE.md         # This file
├── features.txt             # List of model features
├── data/
│   └── heart_2020_cleaned second.csv
├── model/
│   ├── soft_voting_model.pkl    # Soft Voting classifier
│   ├── stacking_model.pkl       # Stacking classifier
│   └── feature_names.pkl        # Feature names list
└── venv/                    # Virtual environment
```

---

_Created for Heart Disease Prediction System v1.0_
