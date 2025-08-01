import tensorflow as tf

class DiceLoss(tf.keras.losses.Loss):
  """Custom Dice Loss Implementation in Keras
  The Dice loss is computed as: 1 - dice_coefficient
    where dice_coefficient = (2 * intersection + smooth) / (sum_of_squares + smooth)

    Args:
        smooth (float): Smoothing factor to avoid division by zero. Default: 1e-8
        axis (list): Axis or axes along which to sum. Default: [1,2,3] (sum over all axes)
        name (str): Name of the loss function. Default: 'dice_loss'
        """
  def __init__(self,smooth=1e-8,axis=[1,2,3],name='dice_loss',**kwargs):
    super().__init__(**kwargs)
    self.smooth=smooth
    self.axis=axis

  def call(self,y_true,y_pred):
    """
        Compute the Dice loss.

        Args:
            y_true: Ground truth labels (same shape as y_pred)
            y_pred: Predicted labels/probabilities (same shape as y_true)

        Returns:
            Dice loss value
        """
    y_true = tf.cast(y_true, y_pred.dtype)
     # Calculate intersection and union
    intersection = tf.reduce_sum(y_true * y_pred, axis=self.axis)
    sum_of_squares = tf.reduce_sum(y_true, axis=self.axis) + tf.reduce_sum(y_pred, axis=self.axis)

        # Calculate Dice coefficient
    dice_coeff = (2. * intersection + self.smooth) / (sum_of_squares + self.smooth)

    return 1.0 - dice_coeff

  def get_config(self):
        """Return the config of the loss function."""
        config = super().get_config()
        config.update({
            'smooth': self.smooth,
            'axis': self.axis,
        })
        return config