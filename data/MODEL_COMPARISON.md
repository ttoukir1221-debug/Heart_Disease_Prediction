# Model Performance Summary & Comparison

## Heart Disease Prediction System

---

## 📊 Model Performance Overview

### Models Evaluated

1. **Stacking Ensemble** - Trained on hybrid-resampled data (ADASYN + SMOTE)
2. **Soft Voting Classifier** - Trained on original data (no sampling)

---

## 📈 Performance Metrics Comparison

| Metric                        | Stacking (Hybrid) | Soft Voting (No Sampling) | Winner      |
| ----------------------------- | ----------------- | ------------------------- | ----------- |
| **Training Accuracy**         | 89.03%            | 91.21%                    | Soft Voting |
| **Test Accuracy**             | 85.14%            | 91.13%                    | Soft Voting |
| **ROC-AUC Score**             | 0.8002            | 0.8342                    | Soft Voting |
| **Recall (Heart Disease)**    | **41%**           | 7%                        | ⭐ Stacking |
| **Precision (Heart Disease)** | 28%               | 58%                       | Soft Voting |
| **F1-Score (Heart Disease)**  | **0.33**          | 0.12                      | ⭐ Stacking |

---

## 🔢 Detailed Classification Reports

### Stacking Ensemble (Hybrid-Sampled)

```
              precision    recall  f1-score   support

           0       0.94      0.90      0.92     54892
           1       0.28      0.41      0.33      5452

    accuracy                           0.85     60344
   macro avg       0.61      0.65      0.62     60344
weighted avg       0.88      0.85      0.86     60344
```

### Soft Voting (No Sampling)

```
              precision    recall  f1-score   support

           0       0.91      1.00      0.95     54892
           1       0.58      0.07      0.12      5452

    accuracy                           0.91     60344
   macro avg       0.75      0.53      0.54     60344
weighted avg       0.88      0.91      0.88     60344
```

---

## 🎯 Confusion Matrix Analysis

### Stacking Ensemble

```
                    Predicted
                 |  No (0)  |  Yes (1) |
Actual  No (0)   |  49,151  |   5,741  |
        Yes (1)  |   3,223  |   2,229  |
```

| Outcome             | Count  | Meaning                                |
| ------------------- | ------ | -------------------------------------- |
| **True Negatives**  | 49,151 | Correctly identified healthy patients  |
| **True Positives**  | 2,229  | ✅ Correctly detected heart disease    |
| **False Positives** | 5,741  | Healthy patients flagged (extra tests) |
| **False Negatives** | 3,223  | ❌ Missed heart disease patients       |

**Detection Rate:** 2,229 / 5,452 = **40.9%** of heart disease cases detected

---

### Soft Voting (No Sampling)

```
                    Predicted
                 |  No (0)  |  Yes (1) |
Actual  No (0)   |  54,627  |     265  |
        Yes (1)  |   5,085  |     367  |
```

| Outcome             | Count  | Meaning                                |
| ------------------- | ------ | -------------------------------------- |
| **True Negatives**  | 54,627 | Correctly identified healthy patients  |
| **True Positives**  | 367    | ✅ Correctly detected heart disease    |
| **False Positives** | 265    | Healthy patients flagged (extra tests) |
| **False Negatives** | 5,085  | ❌ Missed heart disease patients       |

**Detection Rate:** 367 / 5,452 = **6.7%** of heart disease cases detected

---

## ⚖️ Key Differences Explained

### Why Stacking is Better for Medical Use

| Factor                                               | Stacking | Soft Voting | Impact                                     |
| ---------------------------------------------------- | -------- | ----------- | ------------------------------------------ |
| **Patients correctly identified with heart disease** | 2,229    | 367         | Stacking detects **6x more**               |
| **Patients missed (could be fatal)**                 | 3,223    | 5,085       | Soft Voting misses **1,862 more** patients |
| **False alarms**                                     | 5,741    | 265         | Stacking has more false alarms             |

### Trade-off Analysis

```
Cost of False Negative (Missing heart disease):
- Patient goes undiagnosed
- Heart disease progresses
- Potentially fatal outcome
- VERY HIGH COST ⚠️

Cost of False Positive (False alarm):
- Patient gets additional tests
- Minor inconvenience and cost
- Patient is confirmed healthy
- LOW COST ✓
```

---

## 🏆 Final Recommendation

### **WINNER: Stacking Ensemble (Hybrid-Sampled)**

#### Reasons:

1. **6x Better at Detecting Heart Disease**

   - Catches 2,229 vs only 367 patients
   - 41% recall vs 7% recall

2. **Medical Applications Prioritize Recall**

   - Missing a sick patient is worse than a false alarm
   - Better to check 100 people and find 40 cases than check 10 and find 1

3. **Balanced Performance**

   - Uses hybrid sampling (ADASYN + SMOTE) to handle class imbalance
   - Doesn't just predict "No disease" for everyone

4. **Higher F1-Score for Positive Class**
   - 0.33 vs 0.12 (almost 3x better)
   - Better balance of precision and recall

---

## 📉 Why Soft Voting Has Higher Accuracy But Is Worse

The dataset has **class imbalance**:

- 91% patients are healthy (Class 0)
- 9% patients have heart disease (Class 1)

**Soft Voting's strategy:**

- Predict "No disease" for almost everyone
- Gets 91% accuracy (matching the majority class ratio)
- But fails to detect actual heart disease cases

**This is called "accuracy paradox"** - high accuracy but useless predictions!

---

## 🔧 Model Configurations

### Stacking Ensemble

```python
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=6)),
    ('xgb', XGBClassifier(max_depth=5, learning_rate=0.3, n_estimators=100)),
    ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=6))
]
meta_model = LogisticRegression(max_iter=1000)
stack_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)
```

### Soft Voting

```python
soft_voting = VotingClassifier(
    estimators=[
        ('LogisticRegression', LogisticRegression()),
        ('RandomForest', RandomForestClassifier(n_estimators=100, max_depth=6)),
        ('XGBoost', XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=100)),
        ('GradientBoosting', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1))
    ],
    voting='soft'
)
```

---

## 📊 Dataset Information

- **Total samples:** ~301,717 records
- **Test set size:** 60,344 records
- **Features:** 32 (after one-hot encoding)
- **Class distribution:** 91% healthy, 9% heart disease

---

## 📝 Conclusion

| For Your UI        | Recommendation                                       |
| ------------------ | ---------------------------------------------------- |
| **Default Model**  | Stacking Ensemble ⭐                                 |
| **Alternative**    | Soft Voting (if user prefers higher accuracy metric) |
| **Primary Metric** | Recall (for medical applications)                    |

---

_Heart Disease Prediction System v1.0_
_Last Updated: 2026-01-06_
