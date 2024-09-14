import numpy as np
import matplotlib.pyplot as plot


def plot_lines_from_equations(equations):
  """
  Given two linear equations in the a1x + b1y = c format.

  Args:
    equations: A NumPy array of shape (2, 3)
  """

  # Extract coefficients
  a1, b1, c1 = equations[0]
  a2, b2, c2 = equations[1]

  # Calculate slopes and y-intercepts
  m1 = -a1 / b1
  b1 = c1 / b1
  m2 = -a2 / b2
  b2 = c2 / b2

  # Generate x-values
  x = np.linspace(-10, 10, 100)

  # Calculate y-values for each equation
  y1 = m1 * x + b1
  y2 = m2 * x + b2

  # Plot the lines
  plot.plot(x, y1, label=f"y = {m1}x + {b1}")
  plot.plot(x, y2, label=f"y = {m2}x + {b2}")

  # Add labels and a legend
  plot.xlabel("X-axis")
  plot.ylabel("Y-axis")
  plot.title("Linear Equations")
  plot.legend()

  # Show the plot
  plot.show()


# 5X - Y = 10
# 7X + 2Y = 1
equation_coefficients = np.array([[5, -1],
                                 [7, 2]],
                                 dtype=np.dtype(float))

values = np.array([10, 1],
                  dtype=np.dtype(float))

print(f"Shape of Coefficients: {equation_coefficients.shape}")
print(f"Coefficients: {equation_coefficients}")

print(f"Shape of Values: {values.shape}")
print(f"Values: {values}")

#solution
solution_matrix = np.linalg.solve(equation_coefficients, values)

print(f"Solution: {solution_matrix}")

#plot the equations

values_t = values.reshape((2, 1))
equations_matrix = np.hstack((equation_coefficients, values_t))

print(f"Shape of Appended Matrix: {equations_matrix.shape}")
print(equations_matrix[0])
print(equations_matrix[1])

# a1X + b1Y = C1
# a2X + b2Y = C2
plot_lines_from_equations(equations_matrix)
