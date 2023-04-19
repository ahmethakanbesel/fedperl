import matplotlib.pyplot as plt
import numpy as np
from src.modules.settings import DATASET

# Define the number of clients and their respective data counts
clients = ['client0', 'client1', 'client2', 'client3', 'client4', 'client5', 'client6', 'client7', 'client8', 'client9']
labeled_counts = []
unlabeled_counts = []
validation_counts = []

# Loop through each client and generate sample validation data counts
for i, client in enumerate(clients):
    labeled, unlabeled, validation = DATASET.get_client_data_counts(i)
    labeled_counts.append(labeled)
    unlabeled_counts.append(unlabeled)
    validation_counts.append(validation)

# Create stacked bar chart
fig, ax = plt.subplots()
width = 0.35

# Plot labeled data counts
ax.bar(clients, labeled_counts, width, label='Labeled Data')

# Plot unlabeled data counts
ax.bar(clients, unlabeled_counts, width, bottom=labeled_counts, label='Unlabeled Data')

# Plot validation data counts
ax.bar(clients, validation_counts, width, bottom=np.array(labeled_counts) + np.array(unlabeled_counts), label='Validation Data')

# Add labels and legend
ax.set_ylabel('Image Counts')
ax.set_xlabel('Clients')
#ax.set_title('Data Distribution Among the Clients')
ax.legend()

# Show the plot
plt.show()
