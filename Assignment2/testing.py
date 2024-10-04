import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dfd = pd.read_csv("data_problem2.csv")
print(dfd.info())
print(type(dfd))
print(dfd.shape)

file_path = "data_problem2.csv"

# Read the file and process each line
with open(file_path, "r") as f:
    lines = f.readlines()


cleaned_data = dfd.applymap(lambda x: float(str(x).replace("[", "").replace("]", "").strip()))

data_array = cleaned_data.to_numpy()
print(f"data array {type(data_array)}, shape {data_array.shape}, array {data_array}")


n_values = data_array.shape[1] // 2
x_values = data_array[:, :n_values].flatten()
u_values = data_array[:, n_values:].flatten()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.hist(x_values, bins=30, edgecolor='black')
ax1.set_title('Histogram of x values')
ax1.set_xlabel('Value')
ax1.set_ylabel('Frequency')

ax2.hist(u_values, bins=30, edgecolor='black')
ax2.set_title('Histogram of u values')
ax2.set_xlabel('Value')
ax2.set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('2.png')
plt.show()

# Print some basic statistics
print("\nStatistics for x values:")
print(f"Mean: {np.mean(x_values):.2f}")
print(f"Std: {np.std(x_values):.2f}")
print(f"Min: {np.min(x_values):.2f}")
print(f"Max: {np.max(x_values):.2f}")

print("\nStatistics for u values:")
print(f"Mean: {np.mean(u_values):.2f}")
print(f"Std: {np.std(u_values):.2f}")
print(f"Min: {np.min(u_values):.2f}")
print(f"Max: {np.max(u_values):.2f}")