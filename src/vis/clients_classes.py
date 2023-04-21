import matplotlib.pyplot as plt
import numpy as np
from src.modules.settings import DATASET

# Define the number of clients and their respective class distributions
clients = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
class_names = DATASET.classes
class_distributions = [[] for i in range(len(class_names))]
# class_distributions = []


# Loop through each client and generate sample validation data counts
for i, client in enumerate(clients):
    client_classes = DATASET.get_client_class_distribution(client_id=i)
    client_dist = []
    for j, label in enumerate(client_classes):
        class_distributions[j].append(client_classes[label])
        client_dist.append(client_classes[label])

# Create stacked bar chart
fig, ax = plt.subplots()
width = 0.25

for i in range(len(class_names)):
    # Plot labeled data counts
    ax.bar(clients, class_distributions[i], label=class_names[i], bottom=np.sum(class_distributions[:i], axis=0))

# Add labels and legend
ax.set_ylabel('Image Counts')
ax.set_xlabel('Clients')
# ax.set_title('Class Distribution Among the Clients')
ax.legend()

# Show the plot
plt.show()
