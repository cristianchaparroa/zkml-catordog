from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization
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
        # Apple Silicon specific configuration
        print("Apple Silicon M1 detected, configuring Metal backend")
        try:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("Mixed precision enabled for Apple Silicon")
        except Exception as e:
            print(f"Warning: Could not enable mixed precision: {e}")
    else:
        # Standard GPU configuration for NVIDIA GPUs
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                print("GPU is configured for optimized training.")
            except RuntimeError as e:
                print(e)

def create_tiny_model(input_shape=(64, 64, 3)):
    """Create a significantly smaller model"""
    input_layer = Input(shape=input_shape, name='input_layer')

    # Reduced number of filters and layers
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Flatten and dense layers with fewer neurons
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    output_layer = Dense(units=1, activation='sigmoid', dtype=tf.float32, name='output_layer')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def init_tiny_model():
    # Reduced image size from 128x128 to 64x64
    img_size = (64, 64)

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=img_size,
        batch_size=32,
        class_mode='binary')

    model_path = 'tiny_model.keras'

    if os.path.exists(model_path):
        tiny_classifier = tf.keras.models.load_model(model_path)
        print("Tiny model loaded from disk.")
    else:
        # Create the tiny model
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

        tiny_classifier.fit(
            training_set,
            steps_per_epoch=steps_per_epoch,
            epochs=5,  # Reduced number of epochs
            validation_data=test_set,
            validation_steps=validation_steps,
            callbacks=[reduce_lr, early_stopping]
        )

        tiny_classifier.save(model_path)

        # Print model size information
        print("Model architecture summary:")
        tiny_classifier.summary()

        # Print model size in MB
        model_size_bytes = os.path.getsize(model_path)
        model_size_mb = model_size_bytes / (1024 * 1024)
        print(f"Tiny model size: {model_size_mb:.2f} MB")

    return tiny_classifier, training_set.class_indices

def predict_image(classifier, class_indices, image_path, img_size=(64, 64)):
    test_image = image.load_img(image_path, target_size=img_size)
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = test_image / 255.0  # Normalize pixel values

    result = classifier.predict(test_image)
    prediction = 'dog' if result[0][0] > 0.5 else 'cat'
    confidence = result[0][0] if result[0][0] > 0.5 else 1 - result[0][0]

    print(f"Prediction for {image_path}: {prediction} (confidence: {confidence:.2f})")
    return prediction, confidence

def print_model_stats(model_path):
    """Print information about the model size"""
    if not os.path.exists(model_path):
        print("Model doesn't exist.")
        return

    # Get model file size
    model_size_bytes = os.path.getsize(model_path)
    model_size_mb = model_size_bytes / (1024 * 1024)

    # Load the model to get parameter count
    model = tf.keras.models.load_model(model_path)
    total_params = model.count_params()

    print(f"Model size: {model_size_mb:.2f} MB")
    print(f"Total parameters: {total_params:,}")

if __name__ == "__main__":
    setup_gpu()
    tiny_classifier, class_indices = init_tiny_model()

    # Print model stats
    print_model_stats('tiny_model.keras')

    # Compare with original model if it exists
    if os.path.exists('my_model.keras'):
        print("\nComparing with original model:")
        print_model_stats('my_model.keras')

    # Test predictions
    predict_image(tiny_classifier, class_indices, 'dataset/single_prediction/cat_or_dog_1.jpg')
    predict_image(tiny_classifier, class_indices, 'dataset/single_prediction/cat_or_dog_2.jpg')
