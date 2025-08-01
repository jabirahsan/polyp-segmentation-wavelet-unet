import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
from dataset_creator import create_segmentation_dataset
import seaborn as sns
from losses import DiceLoss
from metrics import IoU
from models.wiunet import *
import matplotlib.pyplot as plt

def evaluate_models(saved_model_root_dir='./Final'):
    _,_,test_dataset=create_segmentation_dataset()
    try:
        unet_model= tf.keras.models.load_model(f"{saved_model_root_dir}/unet.keras",
                                                        custom_objects={"DiceLoss": DiceLoss,
                                                                        'IoU':IoU})
        resunetplus_model = tf.keras.models.load_model(f"{saved_model_root_dir}/resunetplus.keras",
                                                        custom_objects={"DiceLoss": DiceLoss,
                                                                        'IoU':IoU})
        loaded_custom=tf.keras.models.load_model(
            f"{saved_model_root_dir}/customnet.keras",
        custom_objects={
            'DWTPooling':DWTPooling,
            'InceptionEncoder':InceptionEncoder,
            'InceptionDecoder':InceptionDecoder,
            'ASPP_Block':ASPP_Block,
            'DiceLoss':DiceLoss,
            'IoU':IoU
        }
        )
    except:
        raise FileNotFoundError("Cannot Locate file for trained models in the path specified")
    
    
    # Test Set Evaluatio

    test_masks=[]

    for images,masks in test_dataset:
        for _,mask in zip(images,masks):
            mask=mask.numpy()
            test_mask_flat = mask.ravel()
            test_masks.extend(test_mask_flat)

    predictions_custom=unet_model.predict(test_dataset,verbose=0)
    binary_prediction = (predictions_custom > 0.5).astype(np.uint8)

    binary_prediction_flat = binary_prediction.ravel()

    report_custom=classification_report(test_masks,binary_prediction_flat)
    print('='*15+'Model Evaluation for UNet')
    print(report_custom)

    conf_custom=confusion_matrix(test_masks,binary_prediction_flat)

    # Plot the confusion matrix as a heatmap
    plt.figure(figsize=(12, 8))
    sns.set_theme(font_scale=1.2)
    sns.heatmap(conf_custom, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Background", "Foreground"], yticklabels=["Background", "Foreground"])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix UNet")
    plt.show()
    
    predictions_custom=resunetplus_model.predict(test_dataset,verbose=0)
    binary_prediction = (predictions_custom > 0.5).astype(np.uint8)

    binary_prediction_flat = binary_prediction.ravel()

    report_custom=classification_report(test_masks,binary_prediction_flat)
    print('='*15+'Model Evaluation for ResUNetPlus')
    print(report_custom)

    conf_custom=confusion_matrix(test_masks,binary_prediction_flat)

    # Plot the confusion matrix as a heatmap
    plt.figure(figsize=(12, 8))
    sns.set_theme(font_scale=1.2)
    sns.heatmap(conf_custom, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Background", "Foreground"], yticklabels=["Background", "Foreground"])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix ResUnetPlus")
    plt.show()

    predictions_custom=loaded_custom.predict(test_dataset,verbose=0)
    binary_prediction = (predictions_custom > 0.5).astype(np.uint8)

    binary_prediction_flat = binary_prediction.ravel()

    report_custom=classification_report(test_masks,binary_prediction_flat)
    print('='*15 + 'Model Evaluation for Custom UNet')
    print(report_custom)

    conf_custom=confusion_matrix(test_masks,binary_prediction_flat)

    # Plot the confusion matrix as a heatmap
    plt.figure(figsize=(12, 8))
    sns.set_theme(font_scale=1.2)
    sns.heatmap(conf_custom, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Background", "Foreground"], yticklabels=["Background", "Foreground"])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix Custom")
    plt.show()