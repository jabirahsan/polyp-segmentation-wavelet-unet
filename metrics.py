import tensorflow as tf

def IoU(y_true, y_pred,smooth=1e-15):
  """Custom IoU merics Implementation
    The IoU is computed as: IoU = true_positives / (true_positives + false_positives + false_negatives)

    Args:
        y_true (Array): Array of ground trruth
        y_pred (Array): Array of predicted values
        smooth (float): Smoothing factor to avoid division by zero. Default: 1e-6
        """
  y_true = tf.cast(y_true,y_pred.dtype)
  intersection =tf.reduce_sum(y_true * y_pred)
  union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
  iou_score = (intersection + smooth) / (union + smooth)
  return iou_score