"""Flask web application for PCA-based customer analysis."""
from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import pickle
import io, base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load models and data
scaler = pickle.load(open("scaler.pkl", "rb"))
pca = pickle.load(open("pca.pkl", "rb"))
clf = pickle.load(open("classifier.pkl", "rb"))
customers = pd.read_csv("customers.csv")

FEATURES = ["TotalSpend", "PurchaseFrequency", "AvgSpend", "CategoryDiversity", "Recency"]
COLORS = {"Budget": "#3498db", "Impulsive": "#e74c3c", "Premium": "#2ecc71"}

RECOMMENDATIONS = {
    "Budget": {
        "offers": ["10% off on bulk purchases", "Free shipping on orders over $30", "Weekly discount coupons"],
        "strategy": "To move from Budget to Premium, gradually increase spending per order and explore more product categories."
    },
    "Impulsive": {
        "offers": ["Flash sale: 25% off for next 2 hours!", "Limited edition bundles", "Buy 2 get 1 free on trending items"],
        "strategy": "To move from Impulsive to Premium, increase purchase frequency and maintain consistent high spending."
    },
    "Premium": {
        "offers": ["Exclusive VIP early access to new collections", "Luxury gift wrapping included", "Personal shopping assistant"],
        "strategy": "You're already a Premium customer! Maintain your engagement to unlock loyalty rewards."
    },
}

def make_scatter_plot():
    X = customers[FEATURES].values
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, color in COLORS.items():
        mask = customers["Label"] == label
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=label, alpha=0.7, edgecolors="w", s=60)
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("Customer Segments (PCA)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

@app.route("/")
def index():
    plot_b64 = make_scatter_plot()
    dist = customers["Label"].value_counts(normalize=True).mul(100).round(1).to_dict()
    X_scaled = scaler.transform(customers[FEATURES].values)
    X_pca = pca.transform(X_scaled)
    points = []
    for i, row in customers.iterrows():
        points.append({
            "id": int(row["Customer ID"]),
            "label": row["Label"],
            "pc1": round(float(X_pca[i, 0]), 3),
            "pc2": round(float(X_pca[i, 1]), 3),
            "features": {f: round(float(row[f]), 2) for f in FEATURES},
        })
    return render_template("index.html", plot=plot_b64, distribution=dist, points=points)

@app.route("/api/group_avg/<group>")
def group_avg(group):
    subset = customers[customers["Label"] == group]
    if subset.empty:
        return jsonify({"error": "No data"}), 404
    avgs = {f: round(float(subset[f].mean()), 2) for f in FEATURES}
    return jsonify(avgs)

@app.route("/api/simulate", methods=["POST"])
def simulate():
    data = request.json
    values = np.array([[data[f] for f in FEATURES]])
    scaled = scaler.transform(values)
    pca_vals = pca.transform(scaled)
    pred = clf.predict(pca_vals)[0]
    rec = RECOMMENDATIONS.get(pred, {})
    return jsonify({
        "predicted": pred,
        "pc1": round(float(pca_vals[0, 0]), 3),
        "pc2": round(float(pca_vals[0, 1]), 3),
        "offers": rec.get("offers", []),
        "strategy": rec.get("strategy", ""),
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
