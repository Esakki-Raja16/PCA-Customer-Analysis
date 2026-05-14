"""Generate a synthetic retail transaction dataset."""
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

n_customers = 300
n_records = 5000

customer_ids = np.random.choice(range(10001, 10001 + n_customers), size=n_records)
descriptions = [
    "WHITE HANGING HEART T-LIGHT HOLDER", "REGENCY CAKESTAND 3 TIER",
    "JUMBO BAG RED RETROSPOT", "PARTY BUNTING", "LUNCH BAG RED RETROSPOT",
    "SET OF 3 CAKE TINS PANTRY DESIGN", "WOODEN PICTURE FRAME WHITE FINISH",
    "KNITTED UNION FLAG HOT WATER BOTTLE", "CREAM CUPID HEARTS COAT HANGER",
    "JAM MAKING SET WITH JARS", "RED WOOLLY HOTTIE WHITE HEART",
    "ALARM CLOCK BAKELIKE GREEN", "SPOTTY BUNTING", "BATH BUILDING BLOCK WORD",
    "DOORMAT NEW ENGLAND", "SET 2 TEA TOWELS I LOVE LONDON",
    "VINTAGE BILLBOARD DRINK ME MUG", "PAPER CHAIN KIT 50S CHRISTMAS",
    "BROCADE RING PURSE", "MINI PAINT SET VINTAGE"
]
stock_codes = [f"S{str(i).zfill(4)}" for i in range(len(descriptions))]

base_date = datetime(2024, 1, 1)
rows = []
for i in range(n_records):
    idx = random.randint(0, len(descriptions) - 1)
    rows.append({
        "Invoice": f"INV{str(random.randint(100000,999999))}",
        "StockCode": stock_codes[idx],
        "Description": descriptions[idx],
        "Quantity": random.randint(1, 20),
        "InvoiceDate": (base_date + timedelta(days=random.randint(0, 364))).strftime("%Y-%m-%d"),
        "Price": round(random.uniform(0.5, 50.0), 2),
        "Customer ID": customer_ids[i],
    })

df = pd.DataFrame(rows)
df.to_csv("dataset.csv", index=False)
print(f"Generated dataset.csv with {len(df)} rows and {df['Customer ID'].nunique()} unique customers.")
