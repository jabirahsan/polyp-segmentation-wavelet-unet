import tensorflow as tf
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


# For reproducibility

tf.random.set_seed(42)
np.random.seed(42)


def decode_img(image_path_tensor):

    img_path=image_path_tensor.numpy().decode('utf-8')

    try:

        image=cv2.imread(img_path,cv2.IMREAD_COLOR)

        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    except:

        raise FileNotFoundError(f'Failed to read image from: {img_path}')

    image=image.astype(np.float32)/255.

    return image


def decode_mask(mask_path_tensor):

    mask_path=mask_path_tensor.numpy().decode('utf-8')

    try:

        mask=cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)

        if mask.ndim==2:

            mask=np.expand_dims(mask,axis=-1)

    except:

        raise FileNotFoundError(f'Failed to read mask from: {mask_path}')

    if np.max(mask)==255:

        mask=mask.astype(np.uint8)/255.

    return mask


def load_mask_image(image_path,mask_path,image_size):

    img=tf.py_function(decode_img,[image_path],tf.float32)

    mask=tf.py_function(decode_mask,[mask_path],tf.uint8)

    img.set_shape([None,None,3])

    mask.set_shape([None,None,1])

    img = tf.image.resize(img, [*image_size])


    # Use nearest neighbor for masks to preserve class labels during resizing

    mask = tf.image.resize(mask, [*image_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


    img.set_shape([*image_size, 3])

    mask.set_shape([*image_size, 1])


    return img, mask


def create_segmentation_dataset(root_dir='./data/PNG', saved_path='./dataset', img_size=(256,256), valid_size=0.1, test_size=0.1, 
                                batch_size=16, shuffle_buffer_size=1000):
    
    """This function will create the tf dataset for the first time and save them in the directory specified by the saved_path.
        The created dataset is saved because so that for future use the same images and masks are used for evaluation and training
        Args:
            root_dir (str): The main path where the data is saved. Defalut: ./data
            saved_path (str): directory path where the dataset will be saved. Defalut: ./dataset
            img_size (tuple): The size that the image will be resized to. Default: (256,256)
            valid_size (float): The percentage value of validation set size. Default: 0.1
            test_size (float): The percentage value of test set size. Default: 0.1
            batch_size (int): Batch size of images and masks. Default: 16
            shuffle_buffer_size (int): Buffer size for shuffling the data. Default: 1000
             """

    if os.path.exists(os.path.join(saved_path,'train_dataset') and os.path.join(saved_path,'valid_dataset') and 
                      os.path.join(saved_path,'test_dataset')):
        test_element_spec = (
            tf.TensorSpec(shape=(None,*img_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,*img_size, 1), dtype=tf.uint8))
                
        train_dataset = tf.data.Dataset.load(os.path.join(saved_path,'train_dataset'), element_spec=test_element_spec)
        valid_dataset = tf.data.Dataset.load(os.path.join(saved_path,'valid_dataset'), element_spec=test_element_spec)
        test_dataset = tf.data.Dataset.load(os.path.join(saved_path,'test_dataset'), element_spec=test_element_spec)
        print('Loaded Dataset From Saved Data')

        return train_dataset,valid_dataset,test_dataset
        
    else:

        images_dir = os.path.join(root_dir, 'Original')

        masks_dir = os.path.join(root_dir, 'Ground Truth')




        if not os.path.exists(images_dir) or not os.path.exists(masks_dir):

            raise FileNotFoundError(f"Image or mask directory not found.\nExpected: {images_dir} and {masks_dir}")


        image_paths=[]

        for image_name in os.listdir(images_dir):

            image_paths.append(os.path.join(images_dir,image_name))


        train_path,test_valid=train_test_split(image_paths,test_size=valid_size+test_size,random_state=42)

        valid_path,test_path=train_test_split(test_valid,test_size=test_size/(valid_size+test_size),random_state=42)

            

        train_basenames = {os.path.splitext(os.path.basename(p))[0]: p for p in train_path}

        valid_basenames = {os.path.splitext(os.path.basename(p))[0]: p for p in valid_path}

        test_basenames = {os.path.splitext(os.path.basename(p))[0]: p for p in test_path}

        mask_paths = sorted([os.path.join(masks_dir, fname)

                                for fname in os.listdir(masks_dir)

                                if fname.lower()])

        mask_basenames = {os.path.splitext(os.path.basename(p))[0]: p for p in mask_paths}


        train_image_paths, train_mask_paths = [], []

        valid_image_paths, valid_mask_paths = [], []

        test_image_paths, test_mask_paths = [], []

        paired_image_paths,paired_mask_paths=[],[]

        for basename, img_path in train_basenames.items():

            paired_image_paths.append(img_path)

            paired_mask_paths.append(mask_basenames[basename])


        train_image_paths = paired_image_paths

        train_mask_paths  = paired_mask_paths



        paired_image_paths,paired_mask_paths=[],[]

        for basename, img_path in valid_basenames.items():

            paired_image_paths.append(img_path)

            paired_mask_paths.append(mask_basenames[basename])


            valid_image_paths = paired_image_paths

            valid_mask_paths= paired_mask_paths


        paired_image_paths,paired_mask_paths=[],[]

        for basename, img_path in test_basenames.items():

            paired_image_paths.append(img_path)

            paired_mask_paths.append(mask_basenames[basename])


            test_image_paths = paired_image_paths

            test_mask_paths= paired_mask_paths


        print(f'Training Image-Masks Pair:{len(train_image_paths)}')

        print(f'Validation Image-Masks Pair:{len(valid_image_paths)}')

        print(f'Test Image-Masks Pair:{len(test_image_paths)}')


        train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_mask_paths))

        valid_dataset = tf.data.Dataset.from_tensor_slices((valid_image_paths, valid_mask_paths))

        test_dataset=tf.data.Dataset.from_tensor_slices((test_image_paths,test_mask_paths))


        

        train_dataset = train_dataset.shuffle(shuffle_buffer_size)


        train_dataset = train_dataset.map(lambda img, mask: load_mask_image(img, mask, img_size), num_parallel_calls=tf.data.AUTOTUNE)

        valid_dataset = valid_dataset.map(lambda img, mask: load_mask_image(img, mask, img_size), num_parallel_calls=tf.data.AUTOTUNE)

        test_dataset = test_dataset.map(lambda img, mask: load_mask_image(img, mask, img_size), num_parallel_calls=tf.data.AUTOTUNE)

        train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        valid_dataset = valid_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        test_dataset=test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        train_path=os.path.join(saved_path,'train_dataset')
        test_path=os.path.join(saved_path,'test_dataset')
        valid_path=os.path.join(saved_path,'valid_dataset')
        os.makedirs(train_path,exist_ok=True)
        os.makedirs(valid_path,exist_ok=True)
        os.makedirs(test_path,exist_ok=True)

        tf.data.experimental.save(train_dataset, train_path)
        tf.data.experimental.save(valid_dataset, valid_path)
        tf.data.experimental.save(test_dataset, test_path)

        return train_dataset, valid_dataset,test_dataset 