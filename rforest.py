from data import Dataset
import pandas as pd

dataset = Dataset(df=pd.read_csv('data.csv'),
                  features=["humidity", "temp", "winddir", "sealevelpressure"],
                  target="precip",
                  final_target="rain",
                  binary_out=True,
                  split=.8,
                  drop_outliers=True,
                  normalize="minmax",
                  )

print