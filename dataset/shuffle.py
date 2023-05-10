import csv
import os
import random
from dotenv import load_dotenv

load_dotenv()

# Get a list of all CSV files in the current directory
csv_files = [os.getenv('ISIC2019_CSV')]

# Loop through each CSV file and randomize the row order
for csv_file in csv_files:
    with open(csv_file, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip the header row
        rows = list(reader)
        random.shuffle(rows)

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
