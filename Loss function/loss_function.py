"""import numpy as np
import matplotlib.pyplot as plt



def generate_dataset(row, col):
    
    input_feature =[]
    
    for i in range(row * col):
        input_feature.append(np.random.randn())
    
    return np.array(input_feature).reshape(row, col)

def generate_label(row, col):
    
    return np.random.randint(2, size=(row, col)).reshape(100, )



X = generate_dataset(100, 2)
Y = generate_label(100,1)

plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap="winter", s=100)
plt.savefig("Dataset_scatter_plot.png")
exit()
print(type(X))
print(Y)
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification


X, Y = make_classification(n_samples=100, n_informative=1, n_redundant=0, n_classes=2, n_features=2, n_clusters_per_class=1, random_state=41, hypercube=False, class_sep=10)

def perceptron(x, y):
    w2 = 1
    b = 1
    w1 = 6
    lr = 0.0001

    for i in range(1000):

        for j in range(x.shape[0]):
            z = w1*x[j][0]+w2*x[j][1]+b

            if z*y[j] > 0:
                w1 = w1+lr*(y[j]*x[j][0])
                w2 = w2+lr*(y[j]*x[j][1])
                b = b+lr*y[j]
    return w1, w2, b


w1, w2, b = perceptron(X, Y)


m = -(w1/w2)
c = -(b/w2)



x_input = np.linspace(-3,3,100)
y_input = m*x_input+c
plt.figure(figsize=(10, 6))
plt.plot(x_input, y_input, color = "red", linewidth = 3)
plt.scatter(X[:,0], X[:, 1], c=Y, cmap="winter", s=100)
plt.ylim(-3, 2)
plt.savefig("Line scatter_Plot.png")



