import pickle
from retention_ai import get_retention_strategy

model = pickle.load(
    open("model.pkl", "rb")
)

print("Model Loaded Successfully")