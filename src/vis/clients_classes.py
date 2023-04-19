import matplotlib.pyplot as plt
import numpy as np
from src.modules.settings import DATASET

# Define the number of clients and their respective class distributions
clients = ['client0', 'client1', 'client2', 'client3', 'client4', 'client5', 'client6', 'client7', 'client8', 'client9']
class_names = DATASET.classes
class_distributions = [[] for i in range(len(class_names))]


# Loop through each client and generate sample validation data counts
for i, client in enumerate(clients):
    client_classes = DATASET.get_client_class_distribution(client_id=i)
    client_dist = []
    for j, label in enumerate(client_classes):
        class_distributions[j].append(client_classes[label])
        client_dist.append(client_classes[label])
    #class_distributions.append(client_dist)

print(class_distributions)

# Create stacked bar chart
fig, ax = plt.subplots()
width = 0.35

for i, label in enumerate(class_names):
    # Plot labeled data counts
    if i == 0:
        ax.bar(clients, class_distributions[i], width, label=label)
    else:
        ax.bar(clients, class_distributions[i], width, bottom=class_distributions[i-1], label=label)

# Add labels and legend
ax.set_ylabel('Image Counts')
ax.set_xlabel('Clients')
#ax.set_title('Class Distribution Among the Clients')
ax.legend()

# Show the plot
plt.show()
