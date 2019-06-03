# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets

import os
cwd = os.getcwd()
print(cwd)

df = pd.read_csv("./data/bird_dino_data.csv")
df.head()

# calculate brain to body ratio
df["Brain Body Ratio"] = df["Total Endocranium (cm3)"]/(df["Body Mass (kg)"]*1000)

# calculate cerebrum to total brain ratio
df["Cerebrum Ratio"] = df["Cerebrum (cm3)"]/df["Total Endocranium (cm3)"]

df.head()

# Change "Bird or Dino" column to binary
# Bird: 0, Dino: 1
df["Bird or Dino"].loc[df["Bird or Dino"] == "Bird"] = 0
df["Bird or Dino"].loc[df["Bird or Dino"] == "Dino"] = 1

df.head()

# convert to numpy matrix
data = df.to_numpy()

# set x and y variables
# x is the brain to body ratio and the cerebrum to whole brain ratio
x = data[:,9:]
print(x)
# y is the bird or dino column
y = data[:,1]


n_neighbors = 15
clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
clf.fit(x, y)
