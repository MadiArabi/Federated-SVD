import numpy as np
import math
import matplotlib.pyplot as plt

def generate_data(m=120, p=150, D=2, sigma=20, seed=5022, missing_level=0.3):
    np.random.seed(seed)
    func = lambda c, t: -c / (math.log(t))
    
    # Generate c and y values
    c = np.random.normal(1, 0.25, size=m)
    y = np.exp(-c / D)
    
    # Generate x values
    x = np.zeros((m, p))
    for i in range(m):
        for j in range(1, p + 1):
            interval = j * 0.006
            if interval < y[i]:
                x[i, j - 1] = func(c[i], interval)
    non_zero_elements = x[x != 0]
    sigmaP = np.sqrt(np.sum(non_zero_elements)) / sigma * 500
    
    # Adding noise
    noise = np.random.normal(0, 0.2, size=x.shape)
    x += noise * (x != 0)

    # Plotting
    xx = np.linspace(0, p, p)
    for i in x:
        plt.plot(xx, i)
    plt.show()
    
    # Introduce missing values
    missing_indices = np.random.choice(m * p, size=int(m * p * missing_level), replace=False)
    x.ravel()[missing_indices] = 0
    
    return x, y
