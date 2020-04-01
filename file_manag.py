"""
file management script for CNN fusulinids v2
"""
import glob
import os
import pathlib
from distutils.dir_util import copy_tree
from random import shuffle
from shutil import copyfile
from shutil import rmtree
from shutil import move

import cv2
import numpy as np

import Augmentor


def move_pic(folder_in, folder_out):
    """
    copy all files nested inside folder in to folder out
    :param folder_in: folder with subfolders and data
    :param folder_out: folder where the data will be saved
    :return: no returns
    """
    for path, _, files in os.walk(folder_in):
        for name in files:
            print(f'{os.path.normpath(path).split(os.path.sep)[1:]} --- {pathlib.PurePath(path, name)}')
            subfolders = os.path.normpath(path).split(os.path.sep)[1:]
            new_filename = f'{subfolders[0]}_{subfolders[1]}_{name}'
            copyfile(pathlib.PurePath(path, name), pathlib.PurePath(folder_out, new_filename))


def crop_imgs(folder_in, folder_out):
    """
    crops center square of images in folder in to folder out
    :param folder_in: folder containing images
    :param folder_out: folder to output images
    :return: no returns
    """
    border = 150
    square = np.array([border, border, 1920 - border, 1280 - border])
    imgs = glob.glob(f'{folder_in}{os.sep}*')
    img_name = os.listdir(folder_in)
    for img_in, name in zip(imgs, img_name):
        print(f'cropping {img_in} saving in into {folder_out}{os.sep}{name.split("_")[0]}')
        # Read image
        img = cv2.imread(img_in)
        # check to see if path exists
        if not os.path.exists(os.path.join(folder_out, name.split("_")[0])):
            os.makedirs(os.path.join(folder_out, name.split("_")[0]))
        # crop images based on square defined above then write cropped image in appropriate location
        imcrop = img[square[1]:square[3], square[0]:square[2]]
        cv2.imwrite(f'{folder_out}{os.sep}{name.split("_")[0]}{os.sep}{name}', imcrop)

    # print the number of images in each one of the folders:
    for folder in os.listdir(folder_out):
        files = os.listdir(f'{folder_out}{os.sep}{folder}')
        print(f'folder {folder:30} has {len(files):5d} files')


def split_data(folder_in, fraction_test, fraction_validation):
    """
    splits the data randomly into train, validation, test
    :param folder_in:
    :param fraction_test:
    :param fraction_validation:
    :return:
    """

    # create test folder:
    test_dir = os.path.join(os.path.dirname(folder_in), f"{os.path.split(folder_in)[-1]}_test")
    rmtree(test_dir, ignore_errors=True)
    os.mkdir(test_dir)

    # create validation folder:
    validation_dir = os.path.join(os.path.dirname(folder_in), f"{os.path.split(folder_in)[-1]}_validation")
    rmtree(validation_dir, ignore_errors=True)
    os.mkdir(validation_dir)

    # create  folder:
    train_dir = os.path.join(os.path.dirname(folder_in), f"{os.path.split(folder_in)[-1]}_train")
    rmtree(train_dir, ignore_errors=True)
    os.mkdir(train_dir)

    # get all folders:
    list_classes = os.listdir(folder_in)

    for i_dir in list_classes:

        imgs = os.listdir(os.path.join(folder_in, i_dir))
        num_samples = len(imgs)
        # shuffle images:
        shuffle(imgs)

        # create test folder:
        os.mkdir(os.path.join(test_dir, i_dir))
        # copy data to test folder:
        for ii in range(np.int(np.ceil(fraction_test * num_samples))):
            img = imgs.pop()
            copyfile(os.path.join(folder_in, i_dir, img), os.path.join(test_dir, i_dir, img))

        # create validation folder:
        os.mkdir(os.path.join(validation_dir, i_dir))
        # copy data to validation folder:
        for ii in range(np.int(np.ceil(fraction_validation * num_samples))):
            img = imgs.pop()
            copyfile(os.path.join(folder_in, i_dir, img), os.path.join(validation_dir, i_dir, img))

        # remaining images will go into training:
        # create training folder:
        os.mkdir(os.path.join(train_dir, i_dir))
        # copy data to test folder:
        train_samples = len(imgs)
        for img in imgs:
            copyfile(os.path.join(folder_in, i_dir, img), os.path.join(train_dir, i_dir, img))

        print(f"{i_dir:30} -- Training: ", end=" ")
        print(f"{train_samples:5d} "
              f" -- Validation: {np.int(np.ceil(fraction_validation * num_samples)):5d} "
              f" -- Test: {np.int(np.ceil(fraction_test * num_samples)):5d}")

def data_augment(folder_in, folder_out, num_samples, img_wid=1620, img_hei=980):
    """
    data augmentation
    :param folder_in: folder with data to be sampled (with subfolders)
    :param folder_out: output folder (data augmented folder)
    :param num_samples: number of samples for each one of the subfolders
    :param img_wid: width dimension of augmented image
    :param img_hei: height dimension of augmented image
    :return:
    """
    # create output folder
    rmtree(folder_out, ignore_errors=True)
    os.mkdir(folder_out)

    # get all folders:
    list_classes = os.listdir(folder_in)
    # loop through all the subfolders
    for i_dir in list_classes:
        # create folder for current class:
        os.mkdir(os.path.join(folder_out, i_dir))

        # use data augmentation pipeline
        p = Augmentor.Pipeline(os.path.join(folder_in, i_dir))
        p.rotate180(probability=0.5)
        p.flip_left_right(probability=0.5)
        p.flip_top_bottom(probability=0.5)
        p.rotate(probability=0.5, max_left_rotation=5, max_right_rotation=5)
        p.random_color(probability=0.5, min_factor=0.01, max_factor=0.8)
        p.random_contrast(probability=0.5, min_factor=0.70, max_factor=0.95)
        p.random_brightness(probability=0.5, min_factor=0.70, max_factor=1.1)
        p.crop_centre(probability=0.3, percentage_area=0.95)
        p.resize(probability=1, width=img_wid, height=img_hei)
        p.sample(num_samples)

        # Augmentator creates a folder called "output", move the folder to the desired location
        copy_tree(os.path.join(folder_in, i_dir, "output"), os.path.join(folder_out, i_dir))
        rmtree(os.path.join(folder_in, i_dir, "output"), ignore_errors=True)

def vert_flip(folder_in):
    """
    saves a vertical flip copy of the images in folder in
    :param folder_in: input folder with images
    :return: saves copy vertically flipped
    """
    imgs = glob.glob(f'{folder_in}{os.sep}*')
    img_name = os.listdir(folder_in)
    for img_in, name in zip(imgs, img_name):
        print(f'flipping {img_in} saving in into {folder_in}')
        # Read image
        img = cv2.imread(img_in)
        # top-bottom flip
        img = cv2.flip(img, 0)
        # get filename and extension:
        filename, file_extension = os.path.splitext(img_in)
        cv2.imwrite(f'{filename}_fvert{file_extension}', img)


def rot180(folder_in):
    """
    saves an image rotated 180 degrees
    :param folder_in: input folder with images
    :return: saves copy
    """
    imgs = glob.glob(f'{folder_in}{os.sep}*')
    img_name = os.listdir(folder_in)
    for img_in, name in zip(imgs, img_name):
        print(f'flipping {img_in} saving in into {folder_in}')
        # Read image
        img = cv2.imread(img_in)
        # top-bottom flip
        img = cv2.flip(img, 1)
        img = cv2.flip(img, 0)
        # get filename and extension:
        filename, file_extension = os.path.splitext(img_in)
        cv2.imwrite(f'{filename}_rot180{file_extension}', img)


if __name__ == "__main__":
    folder_in = '../data/foram'
    folder_out = '../data/genus'
    # move_pic(folder_in, folder_out)

    ## after moving the files, I manually rescaled AMNH images trying to make them same scale as the other ones

    folder_in = 'genus_scaled'
    folder_out = 'genus_crop'
    #crop_imgs(folder_in, folder_out)

    # move it to correct location
    #rmtree(os.path.join("cnn_data_application", folder_out), ignore_errors=True)
    #copy_tree(folder_out, os.path.join("cnn_data_application", folder_out))

    folder_in = os.path.join('..', 'data', 'fusulinid', 'genus_crop')
    # split data:
    split_data(folder_in, fraction_test=0.20, fraction_validation=0.10)

    ########
    # augment training data
    folder_in = os.path.join('..', 'data', 'fusulinid', 'genus_crop_train')
    folder_out = os.path.join('..', 'data', 'fusulinid', 'genus_crop_train_aug')
    nsamples = 64
    data_augment(folder_in, folder_out, nsamples)

    # to facilitate training, augment validation data
    folder_in = os.path.join('..', 'data', 'fusulinid', 'genus_crop_validation')
    folder_out = os.path.join('..', 'data', 'fusulinid', 'genus_crop_validation_aug')
    nsamples = 32
    data_augment(folder_in, folder_out, nsamples)
