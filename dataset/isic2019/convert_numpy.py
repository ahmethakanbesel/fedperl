import csv
import os

from dotenv import load_dotenv
from PIL import Image

import numpy as np

load_dotenv()

# [FILE, ROW_LIMIT]
files = [
    [os.getenv('ISIC2019_CSV'), 1000],
]
img_path = os.getenv('ISIC2019_IMAGES')

image_size = (224, 224)  # Specify the desired image size
images = []
labels = []

for file in files:
    with open(file[0], newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        rows_processed = 0
        for r in reader:
            if rows_processed == file[1]:
                rows_processed = 0
            image = Image.open(os.path.join(img_path, r['image'] + '.jpg')).convert('RGB')
            image = image.resize(image_size)
            image = np.array(image)
            if r['MEL'] == "1.0":
                label = 0
            elif r['NV'] == "1.0":
                label = 1
            elif r['BCC'] == "1.0":
                label = 2
            elif r['AK'] == "1.0":
                label = 3
            elif r['BKL'] == "1.0":
                label = 4
            elif r['DF'] == "1.0":
                label = 5
            elif r['VASC'] == "1.0":
                label = 6
            elif r['SCC'] == "1.0":
                label = 7
            elif r['UNK'] == "1.0":
                label = 8
            else:
                label = 'healthy'
            labels.append(label)
            images.append(image)
            rows_processed += 1

np.save('./isic2019_img.npy', np.array(images))
np.save('./isic2019_lbl.npy', np.array(labels))
