import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

model = joblib.load("artifacts/model_trainer/model.joblib")
X = pd.read_csv("artifacts/data_transformation/test.csv").drop(columns=["Total Fare"])

explainer = shap.Explainer(model.predict, X)

sample = X.sample(1, random_state=42)
shap_values = explainer(sample)

shap.plots.waterfall(shap_values[0])
plt.savefig("artifacts/shap_explanation.png")
