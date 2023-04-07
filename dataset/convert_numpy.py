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
            images.append(Image.open(os.path.join(img_path, r['ImageID'])).convert('RGB'))
            if r['epidural'] == "1":
                labels.append('epidural')
            elif r['intraparenchymal'] == "1":
                labels.append('intraparenchymal')
            elif r['intraventricular'] == "1":
                labels.append('intraventricular')
            elif r['subarachnoid'] == "1":
                labels.append('subarachnoid')
            elif r['subdural'] == "1":
                labels.append('subdural')
            else:
                labels.append('healthy')

np.save('./dataset_img.npy', np.array(images))
np.save('./dataset_lbl.npy', np.array(labels))
