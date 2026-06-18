import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nInformation:")
print(df.info())

print("Missing Values:")
print(df.isnull().sum())
print("Duplicate Rows:")
print(df.duplicated().sum())

print(df.info())

print(df["Churn"].value_counts())
sns.countplot(x="Churn", data=df)

plt.title("Customer Churn Distribution")

plt.show()


plt.figure(figsize=(8,5))

sns.boxplot(
    x="Churn",
    y="MonthlyCharges",
    data=df
)

plt.title("Monthly Charges vs Churn")

plt.show()

print(df.groupby("Churn")["MonthlyCharges"].mean())

plt.figure(figsize=(8,5))

sns.boxplot(
    x="Churn",
    y="tenure",
    data=df
)

plt.title("Tenure vs Churn")

plt.show()


plt.figure(figsize=(8,5))

sns.countplot(
    x="Contract",
    hue="Churn",
    data=df
)

plt.title("Contract Type vs Churn")

plt.xticks(rotation=15)

plt.show()