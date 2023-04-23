# Finds not used images in the main file
import csv

main_file = 'clean_train_df.csv'
files = ['testing.csv', 'training.csv']
headers = None
used_images = set()
not_used_images = []

for file in files:
    with open(file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        headers = next(reader)
        for row in reader:
            used_images.add(row['ImageID'])

with open(main_file, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    file_headers = reader.fieldnames
    for row in reader:
        if row['ImageID'] not in used_images and row['any'] == '1':
            labels = [row[k] for k in file_headers[1:]]
            if sum(map(int, labels)) == 2:  # Chose images which belongs to only one class (+1 for any)
                del row['any']
                not_used_images.append(row)

with open('not_used.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=headers)
    writer.writeheader()
    writer.writerows(not_used_images)
