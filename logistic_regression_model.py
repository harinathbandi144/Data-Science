# diabetes_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score

# Load dataset
df = pd.read_csv("diabetes.csv")

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(6,4))
sns.histplot(df['Glucose'], bins=20, kde=True)
plt.title("Glucose Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df['BMI'], bins=20, kde=True)
plt.title("BMI Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title("Age Distribution")
plt.show()

sns.boxplot(x=df['Glucose'])
plt.title("Box plot of Glucose")
plt.show()

sns.boxplot(x=df['BMI'])
plt.title("Box plot of BMI")
plt.show()

sns.boxplot(x=df['Age'])
plt.title("Box plot of Age")
plt.show()

sns.pairplot(df, hue='Outcome', diag_kind='hist', palette="Set2")
plt.show()

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Handle missing values
col_missing_values = ['Pregnancies', 'Glucose', 'BloodPressure', 
                      'SkinThickness', 'Insulin', 'BMI']
df[col_missing_values] = df[col_missing_values].replace([0, None], np.nan)
df[col_missing_values] = df[col_missing_values].apply(pd.to_numeric)

for col in col_missing_values:
    df[col].fillna(df[col].mean(), inplace=True)

print(df[col_missing_values].isnull().sum())

# Features and target
X = df.drop(columns=['Outcome'])
Y = df[['Outcome']]

# Standardize features
SS = StandardScaler()
SS_X = SS.fit_transform(X)
SS_X = pd.DataFrame(SS_X, columns=X.columns)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(SS_X, Y, test_size=0.2, random_state=42)

# Logistic Regression
model = LogisticRegression()
model.fit(X_train, Y_train)

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Accuracy
print("Train Accuracy:", accuracy_score(Y_train, y_pred_train))
print("Test Accuracy:", accuracy_score(Y_test, y_pred_test))

# Confusion Matrix
print("Confusion Matrix:\n", confusion_matrix(Y_test, y_pred_test))

# Probabilities
df["Y_prob"] = model.predict_proba(SS_X)[:, 1]

# ROC Curve
fpr, tpr, thresholds = roc_curve(Y, df["Y_prob"])
rocscore = roc_auc_score(Y, df["Y_prob"])
print("Area under curve score:", np.round(rocscore, 2))

plt.plot(fpr, tpr, color='red')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()