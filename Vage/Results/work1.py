import math
import matplotlib.pyplot as plt
import numpy as np
import csv

def f(x, A=1):
    term1 = 100 * math.sqrt(abs(A - 0.01 * x**2))
    term2 = 0.01 * abs(x + 10)
    return term1 + term2

x_values = np.arange(-15, 5.01, 0.01)
y_values = [f(x) for x in x_values]

data = list(zip(x_values, y_values))

with open('output.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['x', 'f(x)'])  
    writer.writerows(data)

print("x \t f(x)")
for x, y in zip(x_values[:100], y_values[:100]):
    print(f"{x:.2f} \t {y:.5f}")  

plt.figure(figsize=(10, 5))
plt.plot(x_values, y_values, label="f(x)", color="red")
plt.title("График функции f(x) = 100√|1 - 0.01x²| + 0.01|x + 10|, x∈[-15;5]")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend()
plt.show()