import numpy as np
from matplotlib import pyplot as plt

log_path = '../../weights/'
log_file = 'clnts_acclog_Perl_c8True_avgTrue_proxFalse.npy'
# Load the accuracy log from .npy file
accuracies = np.load(log_path + log_file)

# Get number of clients and colors for plotting
num_clients = accuracies.shape[0]
colors = plt.cm.get_cmap("tab10", num_clients)

# Create a bar chart for each client's accuracy with a different color
for i in range(num_clients):
    plt.bar(i, accuracies[i], color=colors(i), label=f"Client {i+1}")

# Set x-axis tick labels to show client numbers
plt.xticks(range(num_clients), [f"C{i+1}" for i in range(num_clients)])

# Set y-axis maximum value to 1.0
plt.ylim(0, 1.0)

# Set plot title, labels, and legend
plt.title("Accuracies of Clients")
plt.xlabel("Client")
plt.ylabel("Accuracy")
# plt.legend()

# Show the plot
plt.show()