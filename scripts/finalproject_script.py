# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree

df = pd.read_csv("../data/bird_dino_data.csv")
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
# print data type
print(type(data))
print(data[0,0])
print(type(data[0,0]))

# create classifier
clf = tree.DecisionTreeClassifier()

# set x and y variables
# x is the brain to body ratio
x = data[:,9:]
# y is the cerebrum ratio
y = data[:,1]

# fit the classifier
clf = clf.fit(x, y)

# plot the decision tree
tree.plot_tree(clf, class_names=["Bird","Dino"])
plt.show()

#Better visualization
from sklearn.tree import export_graphviz
from IPython.display import Image
import pydotplus
import graphviz

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,feature_names = ['Brain Body Ratio', 'Cerebrum Ratio'],class_names=['Bird','Dino'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# Save a png of the decision tree
graph.write_png('bird_dino.png')
Image(graph.create_png())

# More direct code to just plot the tree
dot_data = tree.export_graphviz(clf, out_file=None,
                     feature_names= ['Brain Body Ratio', 'Cerebrum Ratio'],
                      class_names=['Bird','Dino'],
                      filled=True, rounded=True,
                      special_characters=True)
graph = graphviz.Source(dot_data)
graph 
