Parkinson's Disease Prediction using Machine Learning
🧠 Project Overview
This project aims to develop a predictive machine learning model to detect Parkinson’s disease using Support Vector Machines (SVM). Parkinson’s disease is a progressive neurological disorder, and early diagnosis can significantly impact treatment outcomes. By analyzing clinical and biomedical voice data, the model classifies individuals as either having Parkinson’s disease or being healthy.

📊 Dataset
The dataset used is sourced from the UCI Machine Learning Repository and contains a range of biomedical voice measurements from 31 people, 23 of whom had Parkinson’s disease.

Features include:

MDVP:Fo(Hz) – Average vocal fundamental frequency

MDVP:Jitter(%) – Variation in fundamental frequency

MDVP:Shimmer – Variation in amplitude

NHR, HNR – Measures of noise-to-harmonics ratio

DFA – Signal fractal scaling exponent

Spread1, Spread2 – Nonlinear dynamical complexity measures

Status – Target variable (1: Parkinson’s, 0: Healthy)

⚙️ Tools & Technologies
Programming Language: Python

Libraries Used:

Pandas, NumPy – Data manipulation

Scikit-learn – Model building (SVM), preprocessing, and evaluation

Matplotlib, Seaborn – Data visualization

🧪 Methodology
Data Preprocessing

Handled missing values (if any)

Normalized features for better SVM performance

Split dataset into training and test sets

Model Building

Implemented Support Vector Machine (SVM) with appropriate kernel

Tuned hyperparameters using GridSearchCV

Evaluation Metrics

Accuracy

Precision, Recall, F1-score

Confusion Matrix

✅ Results
The SVM model achieved high accuracy and precision, successfully classifying patients with Parkinson's disease. The model demonstrates promising performance in early diagnosis using vocal biomarker analysis.

📌 Key Takeaways
Support Vector Machines are effective for biomedical classification problems

Vocal measurements can serve as strong indicators for Parkinson’s detection

Early prediction models can support healthcare diagnosis and decision-making

🚀 Future Improvements
Test with larger, more diverse datasets for better generalization

Deploy the model via a web or mobile interface for practical use

Integrate other ML algorithms (e.g., Random Forest, XGBoost) for comparison
