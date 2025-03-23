from tensorflow.keras.models import Sequential, Model, load_model
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
        # M1 can use mixed_float16 with Metal plugin
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

def init_model():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')

    model_path = 'my_model.keras'

    if os.path.exists(model_path):
        classifier = load_model(model_path)
        print("Model loaded from disk.")
    else:
        # Build model using Functional API
        input_layer = Input(shape=(128, 128, 3), name='input_layer')
        x = Conv2D(32, (3, 3), activation='relu')(input_layer)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(units=128, activation='relu')(x)
        x = Dropout(0.5)(x)
        output_layer = Dense(units=1, activation='sigmoid', dtype=tf.float32, name='output_layer')(x)
        classifier = Model(inputs=input_layer, outputs=output_layer)

        classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        test_datagen = ImageDataGenerator(rescale=1./255)
        test_set = test_datagen.flow_from_directory(
            'dataset/test_set',
            target_size=(128, 128),
            batch_size=32,
            class_mode='binary'
        )

        steps_per_epoch = training_set.samples // training_set.batch_size
        validation_steps = test_set.samples // test_set.batch_size

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        classifier.fit(
            training_set,
            steps_per_epoch=steps_per_epoch,
            epochs=10,
            validation_data=test_set,
            validation_steps=validation_steps,
            callbacks=[reduce_lr, early_stopping]
        )

        classifier.save(model_path)

    return classifier, training_set.class_indices

def predict_image(classifier, class_indices, image_path):
    test_image = image.load_img(image_path, target_size=(128, 128))
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
    classifier, class_indices = init_model()
    predict_image(classifier, class_indices, 'dataset/single_prediction/cat_or_dog_1.jpg')
    predict_image(classifier, class_indices, 'dataset/single_prediction/cat_or_dog_2.jpg')
