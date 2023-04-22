import numpy as np
from matplotlib import pyplot as plt
from src.modules.settings import DATASET


# Load the accuracy log from .npy file
counts = DATASET.get_global_test_data_distribution()
print(counts)
# Get number of clients and colors for plotting
colors = plt.cm.get_cmap("tab10", DATASET.num_classes)

# Create a bar chart for each client's accuracy with a different color
for i in range(DATASET.num_classes):
    plt.bar(i, counts[i], color=colors(i), label=DATASET.classes[i])

# Set x-axis tick labels to show client numbers
plt.xticks(range(DATASET.num_classes), [f"C{i+1}" for i in range(DATASET.num_classes)])

# Set plot title, labels, and legend
plt.title("Class Distribution of the Test Dataset")
plt.xlabel("Client")
plt.ylabel("Accuracy")
plt.legend()

# Show the plot
plt.show()