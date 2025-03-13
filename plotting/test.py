import matplotlib.pyplot as plt
import numpy as np

# Sample data
categories = ['A', 'B', 'C', 'D', 'E']
mean_values_a = [10, 15, 12, 18, 14]
min_values_a = [8, 12, 10, 14, 13]
max_values_a = [12, 18, 14, 22, 15]

mean_values_b = [9, 14, 11, 17, 13]
min_values_b = [7, 11, 9, 13, 12]
max_values_b = [11, 17, 13, 21, 14]

# Calculate error bars
lower_error_a = np.array(mean_values_a) - np.array(min_values_a)
upper_error_a = np.array(max_values_a) - np.array(mean_values_a)
error_a = [lower_error_a, upper_error_a]

lower_error_b = np.array(mean_values_b) - np.array(min_values_b)
upper_error_b = np.array(max_values_b) - np.array(mean_values_b)
error_b = [lower_error_b, upper_error_b]

x = np.arange(len(categories))  # The label locations
width = 0.35  # The width of the bars

# Create the bar graph for mean values with error bars
fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, mean_values_a, width, yerr=error_a, capsize=5, label='Mean Value A', color='blue')
bars2 = ax.bar(x + width/2, mean_values_b, width, yerr=error_b, capsize=5, label='Mean Value B', color='orange')

# Add line plots for mean values
ax.plot(x - width/2, mean_values_a, color='blue', marker='o', label='Line Plot A')
ax.plot(x + width/2, mean_values_b, color='orange', marker='o', label='Line Plot B')

# Add labels and title
ax.set_xlabel('Categories')
ax.set_ylabel('Values')
ax.set_title('Mean, Min, and Max Values per Category for A and B')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

# Show the plot
plt.savefig('ok.png')
