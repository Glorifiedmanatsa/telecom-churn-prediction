import pandas as pd


df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Initial Shape:", df.shape)


df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")


print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())


df = df.dropna()

print("\nShape After Dropping Missing Values:", df.shape)


df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

print("\nChurn Distribution:")
print(df["Churn"].value_counts())

print("\nData Ready For Modeling ✅")