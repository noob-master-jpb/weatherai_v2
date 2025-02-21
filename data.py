import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("data2.csv")
train_df = df.sample(frac=0.7, random_state=42)  # 70% training
test_df = df.drop(train_df.index)  # Remaining 30% as test


# Load the CSVs
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# features = ["humidity", "temp", "winddir", "sealevelpressure"]

# min_vals = train_df[features].min()
# max_vals = train_df[features].max()

# train_df[features] = (train_df[features] - min_vals) / (max_vals - min_vals)    
# test_df[features] = (test_df[features] - min_vals) / (max_vals - min_vals)

# print(train_df.head())
# print(test_df.head())



# train_df["rain"] = (train_df["precip"] > 0).astype(int)
# test_df["rain"] = (test_df["precip"] > 0).astype(int)


# Drop the 'precip' column from both datasets
# train_df = train_df.drop(columns=["precip"])
# test_df = test_df.drop(columns=["precip"])

print(train_df.head())
print(test_df.head())
# import matplotlib.pyplot as plt
# test_df.to_csv("test.csv", index=False)
# train_df.to_csv("train.csv", index=False)

import matplotlib.pyplot as plt

# Plot the data
plt.figure(figsize=(10, 6))
for rain_value in [0, 1]:
    subset = train_df[train_df["rain"] == rain_value]
    plt.scatter(subset["temp"], subset["humidity"], label=f"Rain: {rain_value}")

plt.xlabel("Temperature")
plt.ylabel("Humidity")
plt.title("Temperature vs Humidity (colored by Rain)")
plt.legend()
plt.show()
