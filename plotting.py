import numpy as np
import matplotlib.pyplot as plt

# Define grid for x and y
x1 = np.linspace(-10, 10, 100)
x2 = np.linspace(-10, 10, 100)
X1, X2 = np.meshgrid(x1, x2)

# Fix y values
y1, y2, y3 = 1, 2, 3

# Define bilinear form
F = np.array([[1, 0], [0, 2], [1,1]])
Z = X1 * (F[0][0]* y1 + F[1][0]*y2 + F[2][0]*y3)+ X2 * (F[0][1] * y1 + F[1][1] * y2 + F[2][1] * y3)  # Bilinear form: f(x, y)

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Z, cmap='viridis')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x, y)')
plt.show()
