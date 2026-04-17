import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")


df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()


df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})


df = df.drop("customerID", axis=1)


for column in df.columns:
    if df[column].dtype == "object":
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])


X = df.drop("Churn", axis=1)
y = df["Churn"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from xgboost import XGBClassifier


model = XGBClassifier(eval_metric="logloss")
model.fit(X_train, y_train)

 
y_probs = model.predict_proba(X_test)[:, 1]


y_pred = (y_probs > 0.35).astype(int)

print("Threshold Adjusted Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
import joblib

joblib.dump(model, "models/churn_model.pkl")
print("Model saved!")
import os
import pandas as pd
import matplotlib.pyplot as plt

# Create outputs folder automatically
os.makedirs("outputs", exist_ok=True)

# Feature Importance
importance = model.feature_importances_
features = X.columns

feat_imp = pd.Series(importance, index=features).sort_values(ascending=False)

# Debug print (VERY IMPORTANT)
print("\nTop 10 Important Features:\n")
print(feat_imp.head(10))

# Plot
plt.figure(figsize=(8,6))
feat_imp.head(10).plot(kind='barh')
plt.title("Top Features Affecting Churn")
plt.xlabel("Importance Score")
plt.tight_layout()

plt.savefig("outputs/feature_importance.png")
plt.show()