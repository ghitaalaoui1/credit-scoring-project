import joblib

artifacts = joblib.load("models/rf_model.joblib")
print(artifacts.keys())

model = artifacts["model"]
print(type(model))
print(model.feature_names_in_)
print(artifacts["feature_columns"])
print(artifacts["threshold"])
