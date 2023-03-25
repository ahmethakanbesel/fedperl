import csv
import os
from zipfile import ZipFile

label_files = ['clean_train_df.csv', 'testing.csv', 'training.csv']
limit = 500
files_to_keep = []

for label_file in label_files:
    with open(label_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        chosen_files = 0
        for row in reader:
            if chosen_files == limit:
                break
            # print(row['ImageID'])
            files_to_keep.append(row['ImageID'])
            chosen_files += 1

with ZipFile('brain.zip', 'r') as zipObj:
    files_in_zip = zipObj.namelist()
    for file_path in files_in_zip:
        file_name = os.path.basename(file_path)
        if file_name in files_to_keep:
            try:
                zipObj.extract(file_path, 'images')
            except:
                continue
