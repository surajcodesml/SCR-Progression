import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers, Model
import os
from datetime import datetime

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")

# Load data
path = '/home/suraj/Data/Duke_WLOA_RL_Annotated/Duke_WLOA_Control.h5'
f = h5py.File(path, 'r')

def preprocess_data(num_samples=500):  # Reduced samples for debugging
    """Load and preprocess data with proper normalization"""
    print(f"Loading {num_samples} samples...")
    
    # Load raw data
    raw_images = f['images'][:num_samples]
    raw_layers = f['layer_maps'][:num_samples]
    
    # Process images: crop and normalize
    processed_images = []
    processed_layers = []
    
    for i in range(num_samples):
        # Crop image to 300x1000
        img = raw_images[i][:300, :]  # Take first 300 rows
        
        # Normalize image to [0, 1]
        img_min, img_max = np.min(img), np.max(img)
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)
        
        # Process layers: ILM (layer 0) and BM (layer 2)
        ilm = raw_layers[i, :, 0]  # Shape: (1000,)
        bm = raw_layers[i, :, 2]   # Shape: (1000,)
        
        # Adjust for cropping (subtract 0 since we start from row 0)
        # Clip to valid range [0, 299]
        ilm = np.clip(ilm, 0, 299)
        bm = np.clip(bm, 0, 299)
        
        # Normalize layer heights to [0, 1] for better training
        ilm_norm = ilm / 299.0
        bm_norm = bm / 299.0
        
        processed_images.append(img.astype(np.float32))
        processed_layers.append(np.stack([ilm_norm, bm_norm], axis=1).astype(np.float32))
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{num_samples}")
    
    # Convert to arrays and add channel dimension
    images = np.array(processed_images)  # Shape: (N, 300, 1000)
    images = np.expand_dims(images, axis=-1)  # Shape: (N, 300, 1000, 1)
    layers = np.array(processed_layers)  # Shape: (N, 1000, 2)
    
    print(f"Images shape: {images.shape}")
    print(f"Layers shape: {layers.shape}")
    print(f"Image range: [{np.min(images):.3f}, {np.max(images):.3f}]")
    print(f"Layer range: [{np.min(layers):.3f}, {np.max(layers):.3f}]")
    
    return images, layers

def create_simple_model(input_shape=(300, 1000, 1)):
    """Create a simple CNN model"""
    inputs = layers.Input(shape=input_shape)
    
    # CNN layers with smaller filters
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)  # 150x500
    
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)  # 75x250
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)  # 37x125
    
    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    
    # Output layer: 2000 values (1000 points x 2 layers)
    outputs = layers.Dense(2000, activation='sigmoid')(x)  # Sigmoid for normalized output
    
    # Reshape to (1000, 2)
    outputs = layers.Reshape((1000, 2))(outputs)
    
    model = Model(inputs, outputs, name='simple_cnn_regression')
    return model

# Load and preprocess data
images, layers = preprocess_data(num_samples=500)

# Split data
train_size = int(0.8 * len(images))
train_images = images[:train_size]
train_layers = layers[:train_size]
val_images = images[train_size:]
val_layers = layers[train_size:]

print(f"Train: {train_images.shape}, Val: {val_images.shape}")

# Create model
model = create_simple_model()

# Compile with appropriate loss and metrics
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

model.summary()

# Setup TensorBoard logging
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = f"logs/retina_segmentation_{timestamp}"

# Callbacks
callbacks = [
    keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=f'models/best_model_{timestamp}.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

# Train model
print("Starting training...")
history = model.fit(
    train_images, train_layers,
    batch_size=8,  # Small batch size
    epochs=20,
    validation_data=(val_images, val_layers),
    callbacks=callbacks,
    verbose=1
)

# Save hyperparameters
hyperparams = {
    'model_type': 'simple_cnn',
    'input_shape': (300, 1000, 1),
    'batch_size': 8,
    'learning_rate': 0.001,
    'epochs': 20,
    'train_samples': len(train_images),
    'val_samples': len(val_images),
    'timestamp': timestamp
}

import json
with open(f'logs/hyperparams_{timestamp}.json', 'w') as f:
    json.dump(hyperparams, f, indent=2)

# Evaluate model
val_loss, val_mae = model.evaluate(val_images, val_layers, verbose=0)
print(f"\nFinal Validation Results:")
print(f"Loss: {val_loss:.6f}")
print(f"MAE: {val_mae:.6f}")

# Make predictions and visualize
def visualize_predictions(num_samples=3):
    """Visualize predictions vs ground truth"""
    predictions = model.predict(val_images[:num_samples], verbose=0)
    
    # Denormalize predictions back to pixel coordinates
    predictions_denorm = predictions * 299.0
    true_layers_denorm = val_layers[:num_samples] * 299.0
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        ax = axes[i]
        
        # Plot image
        ax.imshow(val_images[i, :, :, 0], cmap='gray', aspect='auto')
        
        # Plot ground truth
        x_coords = np.arange(1000)
        ax.plot(x_coords, true_layers_denorm[i, :, 0], 'r-', label='True ILM', linewidth=2)
        ax.plot(x_coords, true_layers_denorm[i, :, 1], 'g-', label='True BM', linewidth=2)
        
        # Plot predictions
        ax.plot(x_coords, predictions_denorm[i, :, 0], 'r--', label='Pred ILM', linewidth=1, alpha=0.8)
        ax.plot(x_coords, predictions_denorm[i, :, 1], 'g--', label='Pred BM', linewidth=1, alpha=0.8)
        
        ax.set_title(f'Sample {i+1}: Predictions vs Ground Truth')
        ax.legend()
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 300)
    
    plt.tight_layout()
    plt.savefig(f'logs/predictions_{timestamp}.png', dpi=150, bbox_inches='tight')
    plt.show()

# Visualize results
visualize_predictions(3)

# Calculate detailed metrics
all_predictions = model.predict(val_images, verbose=0)
all_predictions_denorm = all_predictions * 299.0
all_true_denorm = val_layers * 299.0

ilm_mae = np.mean(np.abs(all_predictions_denorm[:, :, 0] - all_true_denorm[:, :, 0]))
bm_mae = np.mean(np.abs(all_predictions_denorm[:, :, 1] - all_true_denorm[:, :, 1]))

print(f"\nDetailed Metrics (in pixels):")
print(f"ILM MAE: {ilm_mae:.2f} pixels")
print(f"BM MAE: {bm_mae:.2f} pixels")
print(f"Overall MAE: {(ilm_mae + bm_mae)/2:.2f} pixels")

# Save final model
model.save(f'models/final_model_{timestamp}.keras')
print(f"\nModel and logs saved with timestamp: {timestamp}")
print(f"To view TensorBoard: tensorboard --logdir logs")