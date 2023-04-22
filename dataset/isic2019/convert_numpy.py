import csv
import os

from PIL import Image

import numpy as np

# [FILE, ROW_LIMIT]
files = [
    ['ISIC_2019_Training_GroundTruth.csv', 25331],
]
img_path = './ISIC_2019_Training_Input/'

images = []
labels = []

for file in files:
    with open(file[0], newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        rows_processed = 0
        for r in reader:
            if rows_processed == file[1]:
                rows_processed = 0
            images.append(np.array(Image.open(os.path.join(img_path, r['image'] + '.jpg')).convert('RGB')))
            if r['MEL'] == "1.0":
                labels.append(0)
            elif r['NV'] == "1.0":
                labels.append(1)
            elif r['BCC'] == "1.0":
                labels.append(2)
            elif r['AK'] == "1.0":
                labels.append(3)
            elif r['BKL'] == "1.0":
                labels.append(4)
            elif r['DF'] == "1.0":
                labels.append(5)
            elif r['VASC'] == "1.0":
                labels.append(6)
            elif r['SCC'] == "1.0":
                labels.append(7)
            elif r['UNK'] == "1.0":
                labels.append(8)
            else:
                labels.append('healthy')

np.save('./isic2019_img.npy', np.array(images))
np.save('./isic2019_lbl.npy', np.array(labels))
