from data import Dataset
from map import Distance
import pandas as pd


import matplotlib.pyplot as plt
import random as rd

for ij in range(3,10):

    dataset = Dataset(df=pd.read_csv('data.csv'),
                    features=["humidity","sealevelpressure","windspeed","temp","winddir"],
                    target="precip",
                    final_target="rain",
                    binary_out=True,
                    split=.902,
                    drop_outliers=True,
                    # drop_duplicate=True,
                    normalize="minmax",
                    random_state=402,
                    
                    )
    distance = Distance(datapoints=dataset.train_df,
                        features=dataset.features,
                        target=dataset.target,
                        method="manhattan",
                        k=3,

                        )

    total = 0
    correct = 0
    for i in dataset.test_df.iterrows():
        total += 1
        row = list(i[1])

        distance.calculate_distances(test=row[:-1])
        out = distance.get_closest()["rain"]
        temp = out.value_counts()
        result = temp.idxmax()

        if result == row[-1]:
            correct += 1
        # print(f"Actual: {row[-1]} Prediction: {result}")
    print(correct/total)

    # distance.calculate_distances()
    # print(distance.train_df)
    # print(distance.get_closest())



    # # Plot the training dataset
    # plt.scatter(dataset.train_df["humidity"], dataset.train_df["temp"], c=dataset.train_df["rain"], cmap='bwr')
    # plt.xlabel("Humidity")
    # plt.ylabel("Temperature")
    # plt.title("Training Dataset")
    # plt.colorbar(label="Precipitation")
    # plt.show()`