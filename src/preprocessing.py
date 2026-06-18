import pandas as pd

df = pd.read_csv(
    "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

print(df.head())
print(df.dtypes)
print(df["Churn"].unique())
df["Churn"] = df["Churn"].map({
    "Yes":1,
    "No":0
})
print(df["Churn"].head())
print(df["TotalCharges"].dtype)
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print(df["TotalCharges"].isnull().sum())

df.dropna(inplace=True)

print(df.shape)
print(df.select_dtypes(include="object").columns)

df.drop("customerID", axis=1, inplace=True)

df = pd.get_dummies(df, drop_first=True)

print(df.head())

print(df.shape)

# Features and Target

X = df.drop("Churn", axis=1)

y = df["Churn"]

print(X.shape)
print(y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(X_train.shape)
print(X_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

y_prob = model.predict_proba(X_test)

print(y_prob[:5])

risk_scores = pd.DataFrame({
    "Actual": y_test,
    "Prediction": y_pred,
    "Churn_Risk": y_prob[:,1]
})

print(risk_scores.head(10))









import pickle

pickle.dump(
    model,
    open("model.pkl", "wb")
)

print("Model Saved Successfully")