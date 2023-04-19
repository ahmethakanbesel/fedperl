import matplotlib.pyplot as plt
import numpy as np

# Define the number of clients and their respective data counts
clients = ['client0', 'client1', 'client2', 'client3', 'client4', 'client5', 'client6', 'client7', 'client8', 'client9']
labeled_counts = [100, 150, 200, 50, 120, 180, 90, 200, 100, 250]
unlabeled_counts = [200, 250, 100, 150, 180, 300, 250, 100, 200, 150]
validation_counts = [] # List to store sample validation data counts

# Loop through each client and generate sample validation data counts
for i, client in enumerate(clients):
    # Generate sample validation data counts
    validation_counts.append(np.random.randint(1, 30, size=1)[0]) # replace with actual validation data counts

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
ax.set_ylabel('Data Counts')
ax.set_xlabel('Clients')
ax.set_title('Dataset of Each Client')
ax.legend()

# Show the plot
plt.show()
