from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, DepthwiseConv2D, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
import numpy as np
import os
import platform

def setup_gpu():
    """Configure GPU settings based on platform"""
    is_apple_silicon = platform.processor() == 'arm' and platform.system() == 'Darwin'
    if is_apple_silicon:
        print("Apple Silicon detected, configuring Metal backend")
        try:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("Mixed precision enabled for Apple Silicon")
        except Exception as e:
            print(f"Warning: Could not enable mixed precision: {e}")
    else:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                print("GPU is configured for optimized training.")
            except RuntimeError as e:
                print(e)

def create_tiny_model(input_shape=(16, 16, 3)):
    """Create an extremely lightweight model optimized for ZK proofs"""
    input_layer = Input(shape=input_shape, name='input_layer')
    
    # Use a single depthwise separable convolution with very few filters
    x = DepthwiseConv2D(kernel_size=(3, 3), padding='same', 
                        activation='relu', 
                        # Use depth_multiplier=1 for minimal parameters
                        depth_multiplier=1)(input_layer)
    x = Conv2D(4, kernel_size=(1, 1), activation='relu')(x)  # Just 4 filters
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Global average pooling instead of flatten
    x = GlobalAveragePooling2D()(x)
    
    # Single small dense layer (binary classification)
    output_layer = Dense(units=1, activation='sigmoid', dtype=tf.float32, name='output_layer')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def quantize_model_weights(model, bits=8):
    """Quantize model weights to reduce precision needs in ZK circuits"""
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense) or isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            weights = layer.get_weights()
            for i in range(len(weights)):
                # Only quantize weights, not biases
                if i == 0:
                    # Get min and max values
                    min_val = np.min(weights[i])
                    max_val = np.max(weights[i])
                    
                    # Calculate step size
                    step = (max_val - min_val) / (2**bits - 1)
                    
                    if step > 0:  # Avoid division by zero
                        # Quantize: float -> int -> float
                        quantized = np.round((weights[i] - min_val) / step)
                        weights[i] = quantized * step + min_val
            
            # Set quantized weights back to the layer
            layer.set_weights(weights)
    
    return model

def prune_weights(model, pruning_threshold=0.2):
    """Apply aggressive magnitude-based weight pruning to the model"""
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense) or isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            weights = layer.get_weights()
            for i in range(len(weights)):
                # Only prune weights, not biases (usually weights[0])
                if i == 0:
                    # Get absolute value of weights
                    abs_weights = np.abs(weights[i])
                    # Calculate threshold
                    threshold = pruning_threshold * np.max(abs_weights)
                    # Set values below threshold to zero
                    mask = abs_weights < threshold
                    weights[i][mask] = 0
            # Set pruned weights back to the layer
            layer.set_weights(weights)
    return model

def init_tiny_model(model_path = 'tiny_model.keras'):
    # Reduced image size to 16x16 (from 32x32)
    img_size = (16, 16)
    
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=20,
        brightness_range=[0.8, 1.2]
    )
    
    training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=img_size,
        batch_size=32,
        class_mode='binary')
    
    
    
    if os.path.exists(model_path):
        tiny_classifier = tf.keras.models.load_model(model_path)
        print("Tiny model loaded from disk.")
    else:
        # Create the tiny model
        print("Creating new tiny model...")
        tiny_classifier = create_tiny_model(input_shape=(img_size[0], img_size[1], 3))
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_set = test_datagen.flow_from_directory(
            'dataset/test_set',
            target_size=img_size,
            batch_size=32,
            class_mode='binary'
        )
        
        steps_per_epoch = training_set.samples // training_set.batch_size
        validation_steps = test_set.samples // test_set.batch_size
        
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        print("Training tiny model...")
        tiny_classifier.fit(
            training_set,
            steps_per_epoch=steps_per_epoch,
            epochs=10,
            validation_data=test_set,
            validation_steps=validation_steps,
            callbacks=[reduce_lr, early_stopping]
        )
        
        # Apply aggressive pruning to reduce model size
        print("Applying weight pruning to further reduce model size...")
        tiny_classifier = prune_weights(tiny_classifier, pruning_threshold=0.2)
        
        # Apply weight quantization
        print("Quantizing weights to reduce precision requirements...")
        tiny_classifier = quantize_model_weights(tiny_classifier, bits=6)
        
        # Save the optimized model
        tiny_classifier.save(model_path)
        
        # Print model information
        print("Model architecture summary:")
        tiny_classifier.summary()
        
        # Print model size in MB
        model_size_bytes = os.path.getsize(model_path)
        model_size_mb = model_size_bytes / (1024 * 1024)
        total_params = tiny_classifier.count_params()
        print(f"Tiny model size: {model_size_mb:.2f} MB")
        print(f"Total parameters: {total_params:,}")
        
        # Calculate sparsity (percentage of zero weights)
        zero_weights = 0
        total_weights = 0
        for layer in tiny_classifier.layers:
            if isinstance(layer, tf.keras.layers.Dense) or isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.DepthwiseConv2D):
                weights = layer.get_weights()[0]
                zero_weights += np.sum(weights == 0)
                total_weights += weights.size
        
        if total_weights > 0:
            sparsity = (zero_weights / total_weights) * 100
            print(f"Model sparsity: {sparsity:.2f}% of weights are zero")
    
    return tiny_classifier, training_set.class_indices

def predict_image(classifier, class_indices, image_path, img_size=(16, 16)):
    test_image = image.load_img(image_path, target_size=img_size)
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = test_image / 255.0  # Normalize pixel values
    
    result = classifier.predict(test_image)
    prediction = 'dog' if result[0][0] > 0.5 else 'cat'
    confidence = result[0][0] if result[0][0] > 0.5 else 1 - result[0][0]
    
    print(f"Prediction for {image_path}: {prediction} (confidence: {confidence:.2f})")
    return prediction, confidence

if __name__ == "__main__":
    setup_gpu()
    tiny_classifier, class_indices = init_tiny_model()
    
    # Test predictions
    print("\nTesting model predictions:")
    predict_image(tiny_classifier, class_indices, 'dataset/single_prediction/cat_or_dog_1.jpg')
    predict_image(tiny_classifier, class_indices, 'dataset/single_prediction/cat_or_dog_2.jpg')
