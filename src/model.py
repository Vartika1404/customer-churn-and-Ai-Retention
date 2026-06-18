import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv(
    "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

df["Churn"] = df["Churn"].map({
    "Yes":1,
    "No":0
})

df["Contract"] = df["Contract"].map({
    "Month-to-month":0,
    "One year":1,
    "Two year":2
})

X = df[
    [
        "tenure",
        "MonthlyCharges",
        "Contract"
    ]
]

y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(
    "Accuracy:",
    accuracy_score(y_test, y_pred)
)






import pickle

pickle.dump(
    model,
    open("model.pkl", "wb")
)

print(" Model Saved Successfully")