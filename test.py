import pandas as pd
print("Test script running")
df = pd.read_csv('Tweets.csv', encoding='utf-8', engine='python')
print(f"Dataset loaded. Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(df.head())
