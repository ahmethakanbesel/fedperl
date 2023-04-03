import csv
import os

from PIL import Image

import numpy as np

# [FILE, ROW_LIMIT]
files = [['not_used.csv', 100], ['testing.csv', 500], ['training.csv', 2000]]
all_images = []
all_labels = []
client_limit = 10
image_ids = []
img_path = './images/'

for file in files:
    with open(file[0], newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        client_id = 0
        row_count = 0
        chosen_rows = []
        for row in reader:
            if client_id == client_limit:
                break
            if row_count == file[1]:
                row_count = 0
                if file[0] == 'not_used.csv':
                    output_file = f"client-{client_id}-V"
                elif file[0] == 'training.csv':
                    output_file = f"client-{client_id}-L"
                elif file[0] == 'testing.csv':
                    output_file = f"client-{client_id}-U"

                # try:
                #     with open('./clients/' + output_file, 'w', newline='') as csv_file:
                #         writer = csv.DictWriter(csv_file, fieldnames=chosen_rows[0].keys())
                #         writer.writeheader()
                #         for data in chosen_rows:
                #             image_ids.append(data['ImageID'])
                #             writer.writerow(data)
                # except IOError:
                #     print("I/O error")

                # Save numpy
                img_filenames = []
                img_labels = []

                for r in chosen_rows:
                    img_filenames.append(r['ImageID'])
                    if r['epidural'] == "1":
                        img_labels.append('epidural')
                    elif r['intraparenchymal'] == "1":
                        img_labels.append('intraparenchymal')
                    elif r['intraventricular'] == "1":
                        img_labels.append('intraventricular')
                    elif r['subarachnoid'] == "1":
                        img_labels.append('subarachnoid')
                    elif r['subdural'] == "1":
                        img_labels.append('subdural')
                    else:
                        img_labels.append('healthy')

                all_images.extend(img_filenames)
                all_labels.extend(img_labels)

                # load the images as a numpy array
                img = np.array(
                    [np.array(Image.open(os.path.join(img_path, fn)).convert('RGB')) for fn in img_filenames])

                # generate example labels
                lbl = np.array(img_labels)

                np.save('./clients/' + output_file + '_img.npy', img)
                np.save('./clients/' + output_file + '_lbl.npy', lbl)

                chosen_rows = []
                client_id += 1

            chosen_rows.append(row)
            row_count += 1

# load the images as a numpy array
img_all = np.array([np.array(Image.open(os.path.join(img_path, fn)).convert('RGB')) for fn in all_images])

# generate example labels
lbl_all = np.array(all_labels)

np.save('./dataset_img.npy', img_all)
np.save('./dataset_lbl.npy', lbl_all)
