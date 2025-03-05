from data import Dataset
import pandas as pd

import matplotlib.pyplot as plt


dataset = Dataset(df=pd.read_csv('data.csv'),
                  features=["humidity", "temp", "winddir"],
                  target="precip",
                  final_target="rain",
                  binary_out=True,
                  split=.8,
                  drop_outliers=True,
                  normalize="minmax",
                  )
print(list(dataset.df.columns))



# # Plot the training dataset
# plt.scatter(dataset.train_df["humidity"], dataset.train_df["temp"], c=dataset.train_df["rain"], cmap='bwr')
# plt.xlabel("Humidity")
# plt.ylabel("Temperature")
# plt.title("Training Dataset")
# plt.colorbar(label="Precipitation")
# plt.show()`