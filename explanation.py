import numpy as np
import tensorflow as tf
from losses import DiceLoss
from metrics import IoU
from dataset_creator import create_segmentation_dataset
import matplotlib.pyplot as plt
from models.wiunet import *

def compute_intermediate_heatmaps(features, input_shape):
    """
    Compute normalized and resized heatmaps from intermediate layer outputs.

    Args:
        features: list of feature maps from layers (h_j)
        input_shape: tuple (H, W) — spatial shape of input image

    Returns:
        heatmaps: list of 2D numpy arrays of shape (H, W)
    """
    heatmaps = []

    for idx, h_j in enumerate(features):
        h_j = tf.convert_to_tensor(h_j)  # (H_j, W_j, C_j)
        C = h_j.shape[-1]

        # Step S3: Weighted sum across channels
        w_j = tf.random.normal((C,), stddev=0.05)
        h_j_weighted = tf.reduce_sum(h_j * w_j, axis=-1)  # (H_j, W_j)

        # Step S4: Normalize
        min_val = tf.reduce_min(h_j_weighted)
        max_val = tf.reduce_max(h_j_weighted)
        h_j_norm = (h_j_weighted - min_val) / (max_val - min_val + 1e-8)

        # Step S5: Resize to input shape
        h_j_resized = tf.image.resize(h_j_norm[..., tf.newaxis], input_shape, method='bilinear')  # (H, W, 1)

        heatmaps.append(tf.squeeze(h_j_resized).numpy())  # (H, W)

    return heatmaps



def unet_explanation(saved_model_path='./Final/unet.keras'):
    
    try:
        unet_model=unet_model = tf.keras.models.load_model(saved_model_path,
                                                            custom_objects={"DiceLoss": DiceLoss,
                                                                            'IoU':IoU})
    except:
        raise FileNotFoundError('Cannot locate file for trained Model in the Path Specified')
    _,_,test_dataset=create_segmentation_dataset()
    image=next(iter(test_dataset))[0][0]
    image=tf.expand_dims(image,axis=0)
    all_layer=['conv2d_1','conv2d_3','conv2d_5','conv2d_7','conv2d_8','conv2d_10','conv2d_12','conv2d_14','conv2d_17','conv2d_18']
    layer_outputs = [layer.output for layer in unet_model.layers if layer.name in all_layer]
    activation_model = tf.keras.Model(inputs=unet_model.input, outputs=layer_outputs)
    input_shape = image.shape[1:3]  # (height, width)
    activations = activation_model.predict(image)
    intermediate_heatmaps = compute_intermediate_heatmaps(activations, input_shape)

    n = len(intermediate_heatmaps)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    # Flatten axes for easy iteration
    axes = axes.flatten()

    for i, (ax, hm) in enumerate(zip(axes, intermediate_heatmaps)):
        im = ax.imshow(hm, cmap='winter_r')
        ax.set_title(f"Layer {i}")
        ax.axis('off')
        
        # Add colorbar for this subplot
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.suptitle("Layerwise Heatmaps", fontsize=16, y=1.02)
    plt.show()
        
        
def resunetplus_explanation(saved_model_path='./Final/resunetplus.keras'):
    try:
        resunetplus_model = tf.keras.models.load_model(saved_model_path,
                                                            custom_objects={"DiceLoss": DiceLoss,
                                                                            'IoU':IoU})
    except:
        raise FileNotFoundError('Cannot locate file for trained Model in the Path Specified')
    
    _,_,test_dataset=create_segmentation_dataset()
    image=next(iter(test_dataset))[0][0]
    image=tf.expand_dims(image,axis=0)
    all_layer_resunetplus=['multiply','multiply_1','multiply_2','add_3','add_4','add_7','multiply_6','multiply_8','multiply_9'
                      ,'activation_22']
    layer_outputs = [layer.output for layer in resunetplus_model.layers if layer.name in all_layer_resunetplus]
    activation_model_resunetplus = tf.keras.Model(inputs=resunetplus_model.input, outputs=layer_outputs)

    activations = activation_model_resunetplus.predict(image,verbose=0)
    input_shape = image.shape[1:3]  # (height, width)
    activations = activation_model_resunetplus.predict(image)
    intermediate_heatmaps = compute_intermediate_heatmaps(activations, input_shape)

    n = len(intermediate_heatmaps)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    # Flatten axes for easy iteration
    axes = axes.flatten()

    for i, (ax, hm) in enumerate(zip(axes, intermediate_heatmaps)):
        im = ax.imshow(hm, cmap='winter_r')
        ax.set_title(f"Layer {i}")
        ax.axis('off')
        
        # Add colorbar for this subplot
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.suptitle("Layerwise Heatmaps", fontsize=16, y=1.02)
    plt.show()
    
    
def custom_model_explanation(saved_path='./Final/customnet.keras'):
    try:
        loaded_custom=tf.keras.models.load_model(
            saved_path,
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
        raise FileNotFoundError("Cannot locate file for trained Model in the Path Specified")
    
    _,_,test_dataset=create_segmentation_dataset()
    image=next(iter(test_dataset))[0][0]
    image=tf.expand_dims(image,axis=0)
    target_layers = ['inception_encoder', 'inception_encoder_1', 'inception_encoder_2', 'inception_encoder_3',
                 'aspp__block',
                 'inception_decoder', 'inception_decoder_1', 'inception_decoder_2', 'inception_decoder_3',
                'aspp__block_1','conv2d_8']

    # Filter outputs
    layer_outputs = [layer.output for layer in loaded_custom.layers if layer.name in target_layers]

    # Create the new model to get intermediate activations
    activation_model = tf.keras.Model(inputs=loaded_custom.input, outputs=layer_outputs)

    activations = activation_model.predict(image,verbose=0)
    input_shape = image.shape[1:3]  # (height, width)
    intermediate_heatmaps = compute_intermediate_heatmaps(activations, input_shape)

    n = len(intermediate_heatmaps)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    # Flatten axes for easy iteration
    axes = axes.flatten()

    for i, (ax, hm) in enumerate(zip(axes, intermediate_heatmaps)):
        im = ax.imshow(hm, cmap='winter_r')
        ax.set_title(f"Layer {i}")
        ax.axis('off')
        
        # Add colorbar for this subplot
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.suptitle("Layerwise Heatmaps", fontsize=16, y=1.02)
    plt.show()
    
        