"""Preprocess dataset, engineer features, train PCA + classifier, save models."""
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("dataset.csv")
df["TotalPrice"] = df["Quantity"] * df["Price"]
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
ref_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

cust = df.groupby("Customer ID").agg(
    TotalSpend=("TotalPrice", "sum"),
    PurchaseFrequency=("Invoice", "nunique"),
    AvgSpend=("TotalPrice", "mean"),
    CategoryDiversity=("StockCode", "nunique"),
    Recency=("InvoiceDate", lambda x: (ref_date - x.max()).days),
).reset_index()

# Better balanced labeling using quantile-based scoring
def assign_label(row):
    spend_q = cust["TotalSpend"].quantile([0.33, 0.66])
    freq_q = cust["PurchaseFrequency"].quantile([0.33, 0.66])
    high_spend = row["TotalSpend"] >= spend_q[0.66]
    low_spend = row["TotalSpend"] <= spend_q[0.33]
    high_freq = row["PurchaseFrequency"] >= freq_q[0.66]
    low_freq = row["PurchaseFrequency"] <= freq_q[0.33]
    
    if high_spend and high_freq:
        return "Premium"
    elif low_spend:
        return "Budget"
    else:
        return "Impulsive"

cust["Label"] = cust.apply(assign_label, axis=1)

features = ["TotalSpend", "PurchaseFrequency", "AvgSpend", "CategoryDiversity", "Recency"]
X = cust[features].values
y = cust["Label"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca_model = PCA(n_components=2)
X_pca = pca_model.fit_transform(X_scaled)

clf = DecisionTreeClassifier(random_state=42, max_depth=5)
clf.fit(X_pca, y)

pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(pca_model, open("pca.pkl", "wb"))
pickle.dump(clf, open("classifier.pkl", "wb"))
cust.to_csv("customers.csv", index=False)

print(f"Trained on {len(cust)} customers.")
print(f"Label distribution:\n{pd.Series(y).value_counts()}")
print(f"PCA explained variance: {pca_model.explained_variance_ratio_}")
print("Models saved: scaler.pkl, pca.pkl, classifier.pkl")
