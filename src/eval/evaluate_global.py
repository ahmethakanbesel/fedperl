import glob

import numpy as np
import openpyxl
import torch
from src.modules import models
from openpyxl import Workbook
from sklearn.metrics import precision_recall_fscore_support
from torch import nn

from data_FL import MyDataset, val_transform

LABEL_MAP = {'epidural': 0, 'intraparenchymal': 1, 'intraventricular': 2,
             'subarachnoid': 3, 'subdural': 4}
lesions_names = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
lesions_indexs = [0, 1, 2, 3, 4]

num_classes = 5
batchSize = 16
data_path = '../../dataset/'
name = 'FedPerl'
RESULTS_FILE = 'All_Scores_Global.xlsx'


def predict(model, loader):
    model.eval()
    y_gt = []
    y_preds = []
    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(loader):
            X = sample_batched[0].type(torch.cuda.FloatTensor)
            img_labels = [LABEL_MAP[label] for label in sample_batched[1]]
            y = torch.tensor(img_labels, dtype=torch.long).to('cuda')
            # y = sample_batched[1].type(torch.cuda.LongTensor)
            y_gt.append(y.cpu())
            y_pred = model(X)
            y_pred = np.argmax(y_pred.cpu(), axis=1)
            y_preds.append(y_pred)
    return np.concatenate(y_gt), np.concatenate(y_preds)


def calculate_scores(y, predictions):
    """
    calculate scores using skitlearn

    Parameters:
        y: ground truths
        predictions: predictions

    Returns:
        scores
    """
    acc = np.mean(np.equal(predictions, y))
    prf3 = precision_recall_fscore_support(y, predictions, average='weighted', zero_division=1)

    return acc, prf3[0], prf3[1], prf3[2]


def prepare_excel(filename: str):
    # Create a new workbook object
    workbook = Workbook()
    # Add a sheet to the workbook
    sheet = workbook.active
    sheet.title = 'Val'
    workbook.create_sheet('Test')

    workbook[workbook.sheetnames[0]]['B1'] = 'Accuracy'
    workbook[workbook.sheetnames[0]]['C1'] = 'Precision'
    workbook[workbook.sheetnames[0]]['D1'] = 'Recall'
    workbook[workbook.sheetnames[0]]['E1'] = 'Support'

    workbook[workbook.sheetnames[1]]['B1'] = 'Accuracy'
    workbook[workbook.sheetnames[1]]['C1'] = 'Precision'
    workbook[workbook.sheetnames[1]]['D1'] = 'Recall'
    workbook[workbook.sheetnames[1]]['E1'] = 'Support'
    # Save the workbook to a file
    workbook.save(filename)


def generate_summary(model_path):
    """
    generate model summary and save it an excel sheet

    Parameters:
        None

    Returns:
        None
    """
    models_list = [model_path]

    model = models.get_model(num_classes)
    num_ftrs = model._fc.in_features
    model._fc = nn.Linear(num_ftrs, num_classes)
    device = torch.device('cuda:0')
    model = nn.DataParallel(model)
    model = model.cuda()
    model = model.to(device)

    prepare_excel(RESULTS_FILE)
    wb = openpyxl.load_workbook(RESULTS_FILE)
    i = 1
    shift = 1
    for name in models_list:
        test_ds, val_ds = get_dataset()

        valid_loader = torch.utils.data.DataLoader(val_ds, batch_size=batchSize, shuffle='False', num_workers=0,
                                                   pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batchSize, shuffle='False', num_workers=0,
                                                  pin_memory=True)

        ws = wb[wb.sheetnames[0]]

        model.load_state_dict(torch.load(name))
        # for i in range(5):
        model.eval()

        y, predictions = predict(model, valid_loader)
        Acc, wPr, wR, wF = calculate_scores(y, predictions)
        cl = 'A' + str(i + shift)
        ws[cl] = name
        cl = 'B' + str(i + shift)
        ws[cl] = Acc
        cl = 'C' + str(i + shift)
        ws[cl] = wF
        cl = 'D' + str(i + shift)
        ws[cl] = wPr
        cl = 'E' + str(i + shift)
        ws[cl] = wR
        cl = 'F' + str(i + shift)

        ws = wb[wb.sheetnames[1]]
        y, predictions = predict(model, test_loader)
        Acc, wPr, wR, wF = calculate_scores(y, predictions)

        cl = 'A' + str(i + shift)
        ws[cl] = name
        cl = 'B' + str(i + shift)
        ws[cl] = Acc
        cl = 'C' + str(i + shift)
        ws[cl] = wF
        cl = 'D' + str(i + shift)
        ws[cl] = wPr
        cl = 'E' + str(i + shift)
        ws[cl] = wR
        cl = 'F' + str(i + shift)
        i += 1

    wb.save(RESULTS_FILE)


def get_dataset():
    img_t = np.load(data_path + f'/testing_img.npy')
    lbl_t = np.load(data_path + f'/testing_lbl.npy')

    img_v = np.load(data_path + f'/clients/client-1-V_img.npy')
    lbl_v = np.load(data_path + f'/clients/client-1-V_lbl.npy')

    test_ds = MyDataset(img_t, lbl_t, val_transform)
    val_ds = MyDataset(img_v, lbl_v, val_transform)

    return test_ds, val_ds


if __name__ == '__main__':
    generate_summary('../../models/GlobPerl_c8True_avgTrue_proxFalse_500r.pt')
