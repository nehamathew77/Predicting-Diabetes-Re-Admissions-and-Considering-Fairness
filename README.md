Predicting 30-Day Readmission in Patients with Diabetes: Model Performance and Fairness Analysis

****
Overview

This project examines the factors associated with 30-day hospital readmission among patients with diabetes, using machine learning techniques and fairness evaluation methods. Hospital readmissions are costly for both patients and the healthcare system, and identifying individuals at high risk allows for targeted care interventions. This project is also grounded in my professional experience working in healthcare support roles where I observed how care access, medication adherence, and social conditions strongly influence patient outcomes. Because these factors often correlate with race and socioeconomic status, I extended the analysis to include **fairness and bias auditing**, evaluating whether predictive performance differs across patient racial groups.

****
Part 1: Topic and Dataset Selection

Chosen Topic: Predicting 30-day readmission in diabetic patients and assessing fairness across racial groups.  
Why This Topic? While working in healthcare, I saw firsthand how chronic disease management is deeply unequal across populations. Diabetes, in particular, often intersects with disparities in follow-up access, medication costs, transportation, and health literacy. Predictive modeling is increasingly used in clinical decision support, but if such models encode structural biases, they may exacerbate rather than alleviate disparities. This motivated a combined focus on model performance and equity.Characteristics: 

Format: Structured, tabular data sourced from the NYC Health database (a reliable, public data source). Preparation: All missing values and duplicate records were removed during initial data cleaning in Python, and the dataset was filtered to focus on the most relevant records.

Dataset Selection: Diabetes 130-US Hospitals (1999–2008) — UCI Machine Learning Repository  
Contains:  
- 100,000+ inpatient encounters  
- Patient demographics  
- Diagnosis codes and treatment history  
- Readmission outcomes (<30, >30, none)

****
Part 2: Dataset Cleaning and Restoration

1. Converted placeholder values (`?`) to `NaN`
2. Removed rows with invalid gender entries
3. Dropped non-informative patient identifiers
4. Created binary target: `readmitted == '<30' → 1`, else `0`

Diagnosis Code Restoration Using Clinical Category Grouping
The dataset provides three ICD-9 diagnosis fields (`diag_1`, `diag_2`, `diag_3`).  
Instead of using the raw codes directly, they were **mapped to higher-level condition categories, reducing noise and improving interpretability:

| Diagnosis Range / Value | Category |
|------------------------|----------|
| 390–459, 785 | Circulatory |
| 460–519, 786 | Respiratory |
| 520–579, 787 | Digestive |
| 250 | Diabetes |
| 800–999 | Injury |
| Else | Other |

****
Part 3: Query Development and Testing

Central Question: Can we predict which diabetic patients will be readmitted within 30 days — and does model performance differ across racial groups?

Models Used
| Logistic Regression | Baseline, interpretable coefficients |
| Random Forest | Nonlinear patterns, ensemble learning |
| XGBoost | Strong performance on structured tabular data |
| Neural Network (MLP) | Tests deep learning capability on tabular input |

Handling Class Imbalance
| `class_weight='balanced'` | Logistic Regression, Random Forest |
| `scale_pos_weight` matching negative:positive ratio | XGBoost |
| Class-weight dictionary in `.fit()` | Neural Network |

Performance Metrics
- ROC-AUC — discriminative ability
- F1 Score — performance on the minority class
- Recall (TPR) — critical in clinical risk detection

****
Part 4: Results

| Logistic Regression | ~0.66 | ~0.28 | Best balance of recall and interpretability |
| Random Forest | ~0.67 | ~0.01 | Model collapsed to predicting majority class |
| Neural Network | ~0.62 | ~0.25 | Moderate performance, less stable |
| XGBoost | ~0.66 | ~0.27 | Strong signal, consistent recall |

****
Part 5: Fairness and Bias Evaluation

Why Fairness?
If a model is more likely to misclassify certain racial groups, then real-world deployment could reinforce unequal care delivery**.

Metrics Used
- True Positive Rate (TPR) — Who gets correctly identified as high-risk
- False Positive Rate (FPR) — Who gets flagged unnecessarily

Key Insight: Models that achieved the best predictive recall (Logistic Regression & XGBoost) also showed moderate racial disparities in TPR and FPR.  
The Neural Network exhibited the largest disparities, indicating that model complexity does not guarantee fairness. Random Forest appeared fair only because it failed to identify positive cases. Overall, due to the abundance on data points on Caucasians vs any other racial group, these models are being trained on skewed data.

****
Part 6: Final Insights

Presentation Structure
1. Motivation: Why readmission prediction matters clinically
2. Dataset & feature engineering (especially diagnosis grouping)
3. Model comparison and imbalance handling
4. Fairness evaluation & implications
5. Recommendations for ethical clinical deployment

Personal Reflection
This project deepened my understanding of:
- The challenges of predictive modeling in healthcare where outcomes are shaped by complex behavioral and socioeconomic factors.
- How imbalanced datasets can make “good-looking” metrics misleading.
- The importance of fairness auditing when applying ML to high-stakes environments.

Future Steps
- Apply threshold optimization to reduce TPR disparity across racial groups.
- Use additional social determinants of health data when available.
- Test calibration-by-subgroup to assess reliability of predicted probabilities.
- Obtain extra data that equally represents all racial groups so as to be able to better train predictive models
