# FedPerl: Semi-Supervised Peer Learning

## Installing Requirements

```console
pip install -r requirements.txt
```

## Preparing Dataset

### Converting Images to Numpy Format

There are three examples in dataset folder which show how you can convert images to .npy format.
You should create two separate files for labels and images.
You can store images and labels in two separate arrays.
Label of `images[i]` should be located at `labels[i]`.
Labels must be integers.
If dataset contains string labels you can create a dictionary to replace them with integers.

### Create Dataset Class

After creating .npy version of your dataset you need to prepare class for the dataset.
You can find example dataset classes under `src/data` folder.
Dataset class must implement abstract Dataset class (`src/data/dataset.py`).
Dataset class contains several functions to return indexes of images for given client or the server.
You should look at `src/data/isic2019.py` for additional implementation details.

### Updating settings.py

After implementing your dataset class you need to reserve a name for your dataset.
`src/modules/settings.py` will load the dataset by getting its name from `.env` file.
If you created a new dataset you need to update the if block in `src/modules/settings.py`

## Setting Environment Variables

`.env.example` files should be copied as `.env`. There are 5 fields in the `.env` file.

| Field Name | Value                                                                                                                         |
|------------|-------------------------------------------------------------------------------------------------------------------------------|
| IMG_PATH   | Stores the full path of .npy file for the images                                                                              |
| LBL_PATH   | Stores the full path of .npy file for the labels                                                                              |
| MODEL      | Defines which architecture will be used (efficientnet, efficientnet_legacy, densenet)                                         |
| MODEL_FILE | Stores the full path of your model file after finishing the training process. This is used while evaluating the global model. |
| DATASET    | Defines which dataset will be used (brain, ham10000, isic2019)                                                                |

## Running FedPerl

You can run `src/train_Perl.py` by using command line, or you can set hyperparameters inside that file and run it as a
Python file without using command line.
Before running the file be sure that folders for storing states, weights and models are created.

## Model Evaluation

### Evaluate Global Model

You can run `src/eval/evaluate_global.py` to evaluate the model defined in `.env` file.
It stores the results in a SQLite database `src/eval/results.db`.

### Evaluate Clients' Models

You can run `src/eval/generate_results.py` to evaluate the model defined in `.env` file.
It stores the results in an Excel workbook `src/eval/All_Scores.xlsx`.

## Plotting

Files under `src/vis` contains various files for plotting data distribution among clients from different aspects.