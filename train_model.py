import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

print("===== TRAINING STARTED =====")

# -------------------------------
# LOAD DATASET
# -------------------------------
data = pd.read_csv("loan_approval_dataset.csv")

# Clean column names
data.columns = data.columns.str.strip().str.lower()

print("\nColumns:\n", data.columns)

# -------------------------------
# HANDLE MISSING VALUES
# -------------------------------
print("\nMissing BEFORE:\n", data.isnull().sum())

# Numeric → mean
num_cols = data.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    data[col] = data[col].fillna(data[col].mean())

# Categorical → mode
cat_cols = data.select_dtypes(include=['object']).columns
for col in cat_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

print("\nMissing AFTER:\n", data.isnull().sum())

# -------------------------------
# CLEAN & ENCODE TARGET
# -------------------------------
data['loan_status'] = data['loan_status'].astype(str).str.strip().str.lower()

data['loan_status'] = data['loan_status'].map({
    'approved': 1,
    'rejected': 0,
    'app': 1,
    'rej': 0
})

print("\nLoan Status Encoded:\n", data['loan_status'].head())

# -------------------------------
# ENCODE OTHER CATEGORICALS
# -------------------------------
le = LabelEncoder()

for col in cat_cols:
    if col != 'loan_status':
        data[col] = le.fit_transform(data[col].astype(str))

# Drop useless column
if 'loan_id' in data.columns:
    data = data.drop('loan_id', axis=1)

print("\nCleaned Data:\n", data.head())

# -------------------------------
# 🔥 VISUALIZATION
# -------------------------------

# Loan Approval Count
plt.figure()
sns.countplot(x='loan_status', data=data)
plt.title("Loan Approval Count (0 = Rejected, 1 = Approved)")
plt.savefig("loan_count.png")

# Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(data.corr(), annot=False)
plt.title("Correlation Heatmap")
plt.savefig("heatmap.png")

print("\nGraphs saved: loan_count.png & heatmap.png")

# -------------------------------
# MODEL TRAINING
# -------------------------------
X = data.drop('loan_status', axis=1)
y = data['loan_status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -------------------------------
# SAVE MODEL
# -------------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("columns.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

print("\n===== MODEL + COLUMNS SAVED SUCCESSFULLY =====")