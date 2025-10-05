import pandas as pd
import numpy as np
import os

# Load raw dataset
df = pd.read_csv(r"C:\Guvi\NutriClass Food Classification Using Nutritional Data\data\raw\synthetic_food_dataset_imbalanced.csv")

# 1. Handle missing values (drop or impute)
df = df.dropna()  # or use df.fillna(method='ffill')

# 2. Remove duplicates
df = df.drop_duplicates()

# 3. Cap outliers (example: using 1st and 99th percentiles)
num_cols = ['Calories', 'Protein', 'Fat', 'Carbs', 'Sugar', 'Fiber', 
            'Sodium', 'Cholesterol', 'Glycemic_Index', 'Water_Content']
for col in num_cols:
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    df[col] = np.clip(df[col], lower, upper)

# 4. Normalize numerical features (0-1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Save processed data
os.makedirs(r"C:\Guvi\NutriClass Food Classification Using Nutritional Data\data\processed", exist_ok=True)
df.to_csv(r"C:\Guvi\NutriClass Food Classification Using Nutritional Data\data\processed\food_data_clean.csv", index=False)
print("✅ Data preprocessing completed.")
