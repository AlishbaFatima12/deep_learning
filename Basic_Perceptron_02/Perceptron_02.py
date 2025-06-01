from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt


X, Y = make_classification(n_samples=100, n_informative=1, n_redundant=0, n_classes=2, n_features=2, n_clusters_per_class=1, random_state=41, hypercube=False, class_sep=10)

plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap="winter", s=100)
plt.savefig("Dataset_scatter_plot.png")

def step(z):
    return 1 if z>0 else 0
def perceptron(x, y):
    
    x = np.insert(x, 0, 1, axis=1)
    weights = np.ones(x.shape[1])
    lr = 0.1
    for i in range(0, 1000):
        
        j = np.random.randint(0,100)
        y_hate = step(np.dot(x[j], weights))
        weights = weights+lr*(y[j]-y_hate)*x[j]
        
        return weights[0], weights[1:]
    

intercept_, coef_ = perceptron(X, Y)
print("intercept_ :", intercept_, "\n"+ "coef_ :", coef_)

m = -(coef_[0]/coef_[1])
b = -(intercept_/coef_[1])

print("m :",m, "\n"+"b :", b)


x_input = x_input = np.linspace(-3,3,100)
y_input = m*x_input+b

plt.figure(figsize=(10, 6))

plt.plot(x_input, y_input, color = "red", linewidth = 3)
plt.scatter(X[:,0], X[:, 1], c=Y, cmap="winter", s=100)
plt.ylim(-3, 2)
plt.savefig("Line scatter_Plot.png")


