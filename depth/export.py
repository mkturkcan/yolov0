
def depth_loss(y_true, y_pred):
    # Constants
    ssim_loss_weight = 0.05
    l1_loss_weight = 1.1
    edge_loss_weight = 0.9
    WIDTH = 256  # Assuming the width is 256, adjust if different

    # Edges
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    weights_x = tf.exp(tf.reduce_mean(tf.abs(dx_true)))
    weights_y = tf.exp(tf.reduce_mean(tf.abs(dy_true)))

    # Depth smoothness
    smoothness_x = dx_pred * weights_x
    smoothness_y = dy_pred * weights_y
    depth_smoothness_loss = tf.reduce_mean(abs(smoothness_x)) + tf.reduce_mean(abs(smoothness_y))

    # Structural similarity (SSIM) index
    ssim_loss = tf.reduce_mean(
        1 - tf.image.ssim(
            y_true, y_pred, max_val=WIDTH, filter_size=7, k1=0.01 ** 2, k2=0.03 ** 2
        )
    )

    # Point-wise depth
    l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))

    # Combine losses
    loss = (
        (ssim_loss_weight * ssim_loss)
        + (l1_loss_weight * l1_loss)
        + (edge_loss_weight * depth_smoothness_loss)
    )

    return loss

model = tf.keras.models.load_model('best_model_big_new_depth_3_new.keras', 
                                        custom_objects={'depth_loss': depth_loss})
input_signature = [tf.TensorSpec([1,128,128,3], tf.float32, name='x')]
# Use from_function for tf functions
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
onnx.save(onnx_model, "onnx_export.onnx")