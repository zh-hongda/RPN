import utils
import os
import numpy as np
import argparse
from PIL import Image
import logging

from sklearn.metrics import auc

def image_shape(filename):
    img = Image.open(filename)
    img_arr = np.asarray(img)
    img_shape = img_arr.shape
    return img_shape

def imagefiles2arrs(filenames, augment=False, z_score=False):
    img_shape = image_shape(filenames[0])
    if len(img_shape) == 3:
        images_arr = np.zeros((len(filenames), img_shape[0], img_shape[1], img_shape[2]), dtype=np.float32)
    elif len(img_shape) == 2:
        images_arr = np.zeros((len(filenames), img_shape[0], img_shape[1]), dtype=np.float32)
    # convert to z score per image
    for file_index in range(len(filenames)):
        im = Image.open(filenames[file_index])
        img = np.array(im)
        images_arr[file_index] = img
    return images_arr

if __name__ == '__main__':
    task = 'SE' # EX, HE, MA, SE
    inferenced_image_home_dir = './IDRiD/predicated_result'
    test_set_ground_truth_home_dir = './IDRiD/groundtruth'

    # inferenced_image_home_dir = './DDR/predicated_result'
    # test_set_ground_truth_home_dir = './DDR/groundtruth'

    predicated_mask_dir = os.path.join(inferenced_image_home_dir, task)
    ground_truth_dir = os.path.join(test_set_ground_truth_home_dir, task)
    predicated_mask_filenames = utils.all_files_under(predicated_mask_dir)
    ground_truth_filenames = utils.all_files_under(ground_truth_dir)

    predicated_all = imagefiles2arrs(predicated_mask_filenames) / 255.
    index_gt = 0
    ground_truth_all = np.zeros(predicated_all.shape)
    for index_predicated in range(len(predicated_mask_filenames)):
        if index_gt < len(ground_truth_filenames) and os.path.basename(predicated_mask_filenames[index_predicated]).replace(".jpg", "") in os.path.basename(ground_truth_filenames[index_gt]):
            ground_truth = imagefiles2arrs(ground_truth_filenames[index_gt:index_gt + 1]).astype(np.uint8)[0, ...]
            ground_truth_all[index_predicated, ...] = ground_truth
            index_gt += 1
    aupr_test = utils.pr_metric(ground_truth_all, predicated_all)
    # np.savetxt('draw/precision_EX_IDRiD.out', precision)
    # np.savetxt('draw/recall_EX_IDRiD.out', recall)

    utils.set_logger(os.path.join(os.path.dirname(__file__), 'train.log'))
    logging.info(f'AUPR on test set: {aupr_test}, using results picture at {predicated_mask_dir}')
    logging.info(f'-----------------------------------------------------')
