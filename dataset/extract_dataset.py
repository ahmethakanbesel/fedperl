import csv
import os
from zipfile import ZipFile

# ['dataset.csv', ROW_LIMIT]
label_files = [
    ['not_used.csv', 1000], ['testing.csv', None], ['training.csv', None]
]
files_to_keep = []

for label_file in label_files:
    with open(label_file[0], newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        chosen_files = 0
        for row in reader:
            if label_file[1] is not None and chosen_files == label_file[1]:
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
