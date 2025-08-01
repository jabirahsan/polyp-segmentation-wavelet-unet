import tensorflow as tf
from models import build_unet
from models import res_unet
from models import ResUnetPlusPlus
from models import build_custom_inceptionunet
from losses import DiceLoss
from metrics import IoU
import argparse
import os
from losses import DiceLoss
from metrics import IoU
from dataset_creator import create_segmentation_dataset
import matplotlib.pyplot as plt
from datetime import datetime



def get_model(name):
    if name.lower() == 'unet':
        model=build_unet()
        return model
    elif name.lower() == 'resunet':
        model= res_unet()
        return model
    elif name.lower() == 'resunetplus':
        builder=ResUnetPlusPlus()
        model=builder.build_model()
        return model
    elif name.lower() =='customnet':
        model=build_custom_inceptionunet()
        return model
    else:
        raise ValueError(f"Model '{name}' is not supported. Choose from: unet, resunet, resunetplus or customnet")


def main(args):
    # Create output directories
    os.makedirs(args.save_dir, exist_ok=True)
    log_dir = os.path.join(args.save_dir, "logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    checkpoint_path = os.path.join(args.save_dir, f"{args.model}.keras")

    # Load datasets (replace with your own logic)
    train_dataset, valid_dataset, test_dataset=create_segmentation_dataset()

    # Load model
    model = get_model(args.model)

    # Compile model
    model.compile(
        loss=DiceLoss(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        metrics=[IoU]
    )

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, monitor='val_loss', patience=5),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=1e-3, restore_best_weights=False),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    ]

    # Train
    history = model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=args.epochs,
        callbacks=callbacks
    )

    print(f"\nTraining completed. Model saved to: {checkpoint_path}")
    acc = history.history['io_u']
    val_acc = history.history['val_io_u']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training IoU')
    plt.plot(epochs, val_acc, 'b', label='Validation IoU')  
    plt.title('Training and validation IoU')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()



    plt.show()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train segmentation models")
    parser.add_argument("--model", type=str, required=True, choices=["unet", "resunet", "resunetplus","customnet"],
                        help="Model to train")
    parser.add_argument("--save_dir", type=str, default="./Final", help="Directory to save model and logs")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer")

    args = parser.parse_args()
    main(args)

    