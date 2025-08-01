import tensorflow as tf
import matplotlib.pyplot as plt
from utils import mask_parse
from models.wiunet import *
from evaluation import evaluate_models
from explanation import *
from dataset_creator import create_segmentation_dataset
from losses import DiceLoss
from metrics import IoU


def main(saved_model_root_dir='./Final'):
    evaluate_models()
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
    
    _,_,test_dataset=create_segmentation_dataset()
    for images,masks in test_dataset.take(1):
        for image,mask in zip(images,masks):
            image_exp=tf.expand_dims(image,axis=0)
            y_unet=unet_model.predict(image_exp,verbose=0)
            y_pred_unet = y_unet[0] > 0.5
            y_resunetplus=resunetplus_model.predict(image_exp,verbose=0)
            y_pred_resunetplus=y_resunetplus[0]>0.5
            y_custom=loaded_custom.predict(image_exp,verbose=0)
            y_pred_custom=y_custom[0]>0.5
            h, w, _ = image.shape
        

            #x = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            white_line = np.ones((h, 10, 3)) * 255.0

            all_images = [
                image, white_line,
                mask_parse(mask), white_line,
                mask_parse(y_pred_unet) * 255.0, white_line,
                mask_parse(y_pred_resunetplus) * 255.0 , white_line,
                mask_parse(y_pred_custom) * 255.0 
            ]

            image_show = np.concatenate(all_images, axis = 1)
            plt.imshow(image_show)
            plt.axis('off')
            plt.show()
            break
        
    unet_explanation()
    resunetplus_explanation()
    custom_model_explanation()
    
    

if __name__=="__main__":
    main()
    
    