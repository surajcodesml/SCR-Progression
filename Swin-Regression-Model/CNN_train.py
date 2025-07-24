import h5py
import numpy as np
import tensorflow as tf
from transformers import TFSwinModel
import datetime
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

print(tf.__version__)

# Create a simple CNN model for layer annotation prediction
def create_layer_annotation_model() -> tf.keras.Model:
    """
    Create a CNN model for layer annotation prediction.
    Returns:
        tf.keras.Model: A Keras model instance.
    """
    input_layer = tf.keras.layers.Input(shape=(224, 224, 1))
    
    # CNN backbone
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    
    # Global average pooling to get feature vector
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Dense layers for regression
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    
    # Output layer: predict 224 points for 2 layers (ILM and BM)
    x = tf.keras.layers.Dense(224 * 2, activation='linear')(x)
    output = tf.keras.layers.Reshape((224, 2))(x)
    
    return tf.keras.Model(inputs=input_layer, outputs=output)

def denormalize_layers(layers, layer_min = 0, layer_max = 224):
    return layers * (layer_max - layer_min) + layer_min

def plot_layer_annotations(model, images, layer_maps, num_samples=5, save_dir=None, model_name="model"):
    import os
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    for idx in range(num_samples):
        img = images[idx]
        true_layers = denormalize_layers(layer_maps[idx], layer_min=0, layer_max=224)
        pred_layers = denormalize_layers(model.predict(img[np.newaxis, ...])[0], layer_min=0, layer_max=224)

        plt.figure(figsize=(8, 5))
        plt.imshow(img[:, :, 0], cmap='gray')
        plt.plot(range(224), true_layers[:, 0], 'g-', label='True ILM')
        plt.plot(range(224), true_layers[:, 1], 'b-', label='True BM')
        plt.plot(range(224), pred_layers[:, 0], 'r--', label='Pred ILM')
        plt.plot(range(224), pred_layers[:, 1], 'm--', label='Pred BM')
        plt.title(f"Sample {idx}: Layer Annotations (Denormalized)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        if save_dir is not None:
            filename = f"{model_name}_sample{idx}.png"
            plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight')
        plt.show()
        plt.close()

if __name__ == "__main__":

    # Create and compile model
    model = create_layer_annotation_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()]
    )

    print("Model created successfully!")
    print(f"Input shape: (224, 224, 1)")
    print(f"Output shape: (224, 2)")
    model.summary()

    with h5py.File('/home/suraj/Git/SCR-Progression/Duke_Control_processed.h5', 'r') as f:
        images = f['images'][:]  # shape: (N, 224, 224)
        layer_maps = f['layer_maps'][:]  # shape: (N, 224, 2) or (N, 224, 3)

    # add another dimension to images for compatibility
    if images.ndim == 3:
        images = np.expand_dims(images, axis=-1)

    # We only want ILM and BM (first and last columns) for training
    layer_maps = layer_maps[:, :, [0, 2]]  # if shape is (N, 224, 3)

    #TEST :: Comment out below when training with full dataset
    #images = images[:1000]
    #layer_maps = layer_maps[:1000]

    X_train, X_test, y_train, y_test = train_test_split(
        images, layer_maps, test_size=0.2, random_state=42
    )

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32).shuffle(1000)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

    # TensorBoard callback
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Train model
    model.fit(train_dataset,
            validation_data=test_dataset, 
            epochs=50, 
            callbacks=[tensorboard_callback])

    # Evaluate model
    test_loss = model.evaluate(test_dataset)
    print(f"Test MSE: {test_loss}")

    #To save the model
    model.save('CNN_regression_model_with_preprocessed_image.h5')

    #TO load the model later
    #model = tf.keras.models.load_model('CNN_regression_model_with_preprocessed_image.h5')
    #model.summary()

    # Visualize predictions on test set
    plot_layer_annotations(
        model, 
        X_test, 
        y_test, 
        num_samples=5, 
        save_dir=log_dir, 
        model_name="CNN_regression_model_with_preprocessed_image"
    )