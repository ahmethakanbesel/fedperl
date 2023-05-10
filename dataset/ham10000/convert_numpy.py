import csv
import os

from dotenv import load_dotenv
from PIL import Image

import numpy as np

load_dotenv()

# [FILE, ROW_LIMIT]
files = [
    [os.getenv('HAM10000_CSV'), 100000],
]
img_path = os.getenv('HAM10000_IMAGES')

image_size = (224, 224)  # Specify the desired image size
images = []
labels = []
dx_ids = {'bkl': 0, 'nv': 1, 'df': 2, 'mel': 3, 'vasc': 4, 'bcc': 5, 'akiec': 6}

for file in files:
    with open(file[0], newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        rows_processed = 0
        for r in reader:
            if rows_processed == file[1]:
                rows_processed = 0
            image = Image.open(os.path.join(img_path, r['image_id'] + '.jpg')).convert('RGB')
            image = image.resize(image_size)
            image = np.array(image)
            if r['dx'] not in dx_ids:
                continue
            else:
                label = dx_ids[r['dx']]
            labels.append(label)
            images.append(image)
            rows_processed += 1

np.save('./ham10000_img.npy', np.array(images))
np.save('./ham10000_lbl.npy', np.array(labels))
