import csv
import os

from PIL import Image

import numpy as np

# [FILE, ROW_LIMIT]
files = [
    ['ISIC_2019_Training_GroundTruth.csv', 25331],
]
img_path = './ISIC_2019_Training_Input/'


def preprocess_images(file_path, row_limit, batch_size, image_size):
    images_shape = (row_limit, image_size[0], image_size[1], 3)
    images = np.memmap('./isic2019_img.npy', dtype=np.uint8, mode='w+', shape=images_shape)
    labels = []
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        rows_processed = 0
        batch_idx = 0
        for r in reader:
            if rows_processed == row_limit:
                break
            image = Image.open(os.path.join(img_path, r['image'] + '.jpg')).convert('RGB')
            image = image.resize(image_size)  # Resize image to specified size
            image = np.array(image)
            if batch_idx >= images.shape[0]:
                break
            images[batch_idx] = image
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
            rows_processed += 1
            batch_idx += 1
            if batch_idx == batch_size:
                batch_idx = 0
    return labels


batch_size = 1000
image_size = (224, 224)  # Specify the desired image size
all_labels = []

for file in files:
    labels = preprocess_images(file[0], file[1], batch_size, image_size)
    all_labels.extend(labels)

np.save('./isic2019_lbl.npy', np.array(all_labels))
