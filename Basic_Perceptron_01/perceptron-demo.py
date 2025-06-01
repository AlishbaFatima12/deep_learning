import numpy as np
import matplotlib as plt
from matplotlib.pyplot import figure
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Perceptron
from mlxtend.plotting import plot_decision_regions
#load dataset
df = pd.read_csv("placement.csv")
#plote loaded dataset and save ot in png format
scatter_plote = sns.scatterplot(data=df, x='cgpa', y='resume_score', hue='placed')
scatter_plote.figure.savefig("scatter_plot.png")
#Make a cordinates X, Y to fit the model by slicing the dataset
X = df.iloc[:, 0:2]
Y = df.iloc[:, -1]

#create an Object of Perceptron and fit the model
P = Perceptron()
P.fit(X.values, Y.values)
# Print weights and biese values . w1, w2 b
print(P.coef_)       #w1 = 40.26, w2 = -36.
print(P.intercept_)  #b = -25.


# Show perceptron decision line graph on the base of perceptron classifier
f = figure()
classifier_decision_graph = plot_decision_regions(X.values, Y.values, clf=P, legend=2)
f.savefig("classifier_decision_graph.png")