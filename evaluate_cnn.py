# Rafael Pires de Lima
# July 2019
# evaluate cnn model
# functions to create confusion matrices, plot all examples with the corresponding label

import os

import numpy as np
import pandas as pd
import seaborn as sns
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from fit_models_experiments import model_preprocess


def label_folder(path_folder, path_model, arch):
    """Labels (classifies) a folder containing subfloders based on a retrained CNN model.
      Args:
        path_folder: String path to a folder containing subfolders of images.
        path_model: String path to the model to be used for classification.
        arch: model name
      Returns:
        List: a numpy array with predictions (pred) and the file names of the images classified (generator.filenames)
      """
    # load the model:
    model = load_model(path_model)

    # get model input parameters:
    img_height = model.layers[0].get_output_at(0).get_shape().as_list()[1]
    img_width = model.layers[0].get_output_at(0).get_shape().as_list()[2]

    datagen = ImageDataGenerator(preprocessing_function=model_preprocess(arch))

    # flow from directory:
    generator = datagen.flow_from_directory(
        path_folder,
        target_size=(img_width, img_height),
        batch_size=1,
        class_mode=None,
        shuffle=False)

    if len(generator) > 0:
        # if data file is structured as path_folder/classes, we can use the generator:
        pred = model.predict_generator(generator, steps=len(generator), verbose=1)
    else:
        # the path_folder contains all the images to be classified
        # TODO: if problems arise
        pass

    # save results as dataframe
    df = pd.DataFrame(pred, columns=generator.class_indices.keys())
    df['file'] = generator.filenames
    df['true_label'] = df['file'].apply(os.path.dirname).apply(str.lower)
    df['pred_idx'] = np.argmax(df[generator.class_indices.keys()].to_numpy(), axis=1)
    # save as the label (dictionary comprehension because generator.class_indices has the
    # key,values inverted to what we want
    df['pred_label'] = df['pred_idx'].map({value: key for key, value in generator.class_indices.items()}).apply(
        str.lower)
    # save the maximum probability for easier reference:
    df['max_prob'] = np.amax(df[generator.class_indices.keys()].to_numpy(), axis=1)
    return df

def plot_res(filenames, y_true, y_pred, y_prob, plot_name="results"):
    """
    uses the list of filenames and the other input to plot the image with the corresponding predicted label
    :param filenames: filenames of images classified
    :param y_true: the true label of the image
    :param y_pred: the predicted label of the image
    :param y_prob: the probability assigned to the image
    :param plot_name: name of the plot to be saved
    :param dpi: dots per inch
    :return:
    """

    # set the figure:
    nrows = 9
    ncols = 4
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 16))
    [ax.ravel()[ii].axis('off') for ii in range(len(ax.ravel()))]
    for idx in range(len(filenames)):

        if not idx % (nrows * ncols) and idx > 0:
            # save image and open a new one
            plt.tight_layout()
            plt.savefig(f"{plot_name}_{idx}.pdf")
            plt.close()
            print(idx)
            fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 16))
            [ax.ravel()[ii].axis('off') for ii in range(len(ax.ravel()))]

        # read the image:
        sample = mpimg.imread(filenames[idx])
        ax.ravel()[idx % (nrows * ncols)].imshow(sample)
        #ax.ravel()[idx % (nrows * ncols)].axis('off')
        ax.ravel()[idx % (nrows * ncols)].set_title(y_true[idx])
        if y_true[idx] == y_pred[idx]:
            bbox = dict(boxstyle="round",
                        ec=(0.5, 1., 0.5),
                        fc=(0.8, 1., 0.8),
                        )
        else:
            bbox = dict(boxstyle="round",
                        ec=(1., 0.5, 0.5),
                        fc=(1., 0.8, 0.8),
                        )
        ax.ravel()[idx % (nrows * ncols)].text(sample.shape[1] // 2, np.int(0.9 * sample.shape[0]),
                                               f"{y_pred[idx]}: {y_prob[idx]:.2f}",
                                               size=9,
                                               ha="center", va="center",
                                               bbox=bbox
                                               )

    # save last figure
    plt.tight_layout()
    plt.savefig(f"{plot_name}_{idx}.pdf")
    plt.close()


if __name__ == "__main__":

    model_dir = "../models/"
    images_dir = '../images/'
    test_dir = "../data/fusulinid/genus_crop_test"

    # models tested
    models_test = {
#        'VGG19_fusulinid_frozen': (299, 299, 3),
#        'InceptionV3_fusulinid_frozen': (299, 299, 3),
#        'MobileNetV2_fusulinid_frozen': (224, 224, 3),
#        'ResNet50_fusulinid_frozen': (299, 299, 3),
#        'DenseNet121_fusulinid_frozen': (299, 299, 3),
#
#        'VGG19_fusulinid_fine_tune': (299, 299, 3),
#        'InceptionV3_fusulinid_fine_tune': (299, 299, 3),
#        'MobileNetV2_fusulinid_fine_tune': (224, 224, 3),
#        'ResNet50_fusulinid_fine_tune': (299, 299, 3),
#        'DenseNet121_fusulinid_fine_tune': (299, 299, 3),
        
        'VGG19_fusulinid_rand_init': (299, 299, 3),
        'InceptionV3_fusulinid_rand_init': (299, 299, 3),
        'MobileNetV2_fusulinid_rand_init': (224, 224, 3),
        'ResNet50_fusulinid_rand_init': (299, 299, 3),
        'DenseNet121_fusulinid_rand_init': (299, 299, 3),


    }

    for model_name in models_test:
        print(model_name)
        pred_df = label_folder(test_dir, os.path.join(model_dir, f"{model_name}.hdf5"), model_name.split('_')[0])
        # save the results:
        pred_df.to_csv(os.path.join(model_dir, f"{model_name}.csv"), index=False)
        # plot the results:
        pred_df['filename'] = pred_df['file'].apply(lambda x: os.path.join(test_dir, f"{x}"))
        plot_res(pred_df['filename'],
                 y_true=pred_df['true_label'],
                 y_pred=pred_df['pred_label'],
                 y_prob=pred_df['max_prob'],
                 plot_name=os.path.join(images_dir, f"res-{model_name}")
                 )

    print("\n\n\ncomplete")
