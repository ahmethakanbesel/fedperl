import csv
import os

from PIL import Image

import numpy as np

# [FILE, ROW_LIMIT]
files = [
    ['validation.csv', 1000],
    ['training.csv', 20000],
    ['testing.csv', 5000],
]
img_path = './images/'

images = []
labels = []

for file in files:
    with open(file[0], newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        rows_processed = 0
        for r in reader:
            if rows_processed == file[1]:
                rows_processed = 0
            images.append(np.array(Image.open(os.path.join(img_path, r['ImageID'])).convert('L')))
            if r['epidural'] == "1":
                labels.append(0)
            elif r['intraparenchymal'] == "1":
                labels.append(1)
            elif r['intraventricular'] == "1":
                labels.append(2)
            elif r['subarachnoid'] == "1":
                labels.append(3)
            elif r['subdural'] == "1":
                labels.append(4)
            else:
                labels.append('healthy')

np.save('./ct_dataset_img.npy', np.array(images))
np.save('./ct_dataset_lbl.npy', np.array(labels))
