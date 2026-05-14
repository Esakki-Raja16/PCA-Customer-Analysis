# PCA Customer Analysis

Dimensionality Reduction and Feature Optimization using Principal Component Analysis for Customer Data.

## Folder Structure

```
pca_project/
├── app.py                 # Flask web server
├── generate_dataset.py    # Creates synthetic dataset
├── train_model.py         # Feature engineering + PCA + classifier training
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html         # Frontend (Bootstrap)
└── README.md
```

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the dataset
python generate_dataset.py

# 3. Train models (scaler, PCA, classifier)
python train_model.py

# 4. Start the web app
python app.py
```

Then open http://localhost:5000 in your browser.

## Features

- **Tab 1 – Automated Analysis**: PCA scatter plot colored by customer type, click any customer to see details and business insights.
- **Tab 2 – Simulation & Recommendation**: Select a group, adjust feature sliders, get real-time predictions and personalized offers.
