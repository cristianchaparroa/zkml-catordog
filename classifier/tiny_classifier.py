from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
import numpy as np
import os
import platform
import matplotlib.pyplot as plt

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

def create_simple_model(input_shape=(64, 64, 3)):
    """Create a simple effective model that's easy to train"""
    model = Sequential([
        # First Conv Block
        Conv2D(16, (3, 3), padding='same', input_shape=input_shape, activation='relu'),
        MaxPooling2D((2, 2)),

        # Second Conv Block
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        MaxPooling2D((2, 2)),

        # Third Conv Block
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D((2, 2)),

        # Classification Head
        GlobalAveragePooling2D(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def simple_preprocess(img):
    """Simple preprocessing that ensures consistent scaling"""
    img = tf.cast(img, tf.float32) / 255.0
    return img

def init_simple_model(model_path='simple_model.keras'):
    # Use a larger image size for better feature detection
    img_size = (64, 64)

    # Simple but effective data augmentation
    train_datagen = ImageDataGenerator(
        preprocessing_function=simple_preprocess,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2  # Create a validation split
    )

    try:
        # Load the data
        training_set = train_datagen.flow_from_directory(
            'dataset/training_set',
            target_size=img_size,
            batch_size=32,
            class_mode='binary',
            subset='training')

        validation_set = train_datagen.flow_from_directory(
            'dataset/training_set',
            target_size=img_size,
            batch_size=32,
            class_mode='binary',
            subset='validation')

        # Check class balance
        print("\nClass distribution in training data:")
        class_counts = np.bincount(training_set.classes)
        for class_id, count in enumerate(class_counts):
            class_name = list(training_set.class_indices.keys())[
                list(training_set.class_indices.values()).index(class_id)
            ]
            print(f"{class_name}: {count} images")

        # Create or load model
        if os.path.exists(model_path):
            try:
                model = tf.keras.models.load_model(model_path)
                print("Simple model loaded from disk.")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Creating a new model instead.")
                model = create_simple_model(input_shape=(img_size[0], img_size[1], 3))
        else:
            print("Creating new simple model...")
            model = create_simple_model(input_shape=(img_size[0], img_size[1], 3))

            try:
                # Configure test data
                test_datagen = ImageDataGenerator(
                    preprocessing_function=simple_preprocess
                )
                test_set = test_datagen.flow_from_directory(
                    'dataset/test_set',
                    target_size=img_size,
                    batch_size=32,
                    class_mode='binary'
                )

                # Calculate steps
                steps_per_epoch = training_set.samples // training_set.batch_size
                validation_steps = validation_set.samples // validation_set.batch_size
                test_steps = test_set.samples // test_set.batch_size

                # Configure callbacks
                early_stopping = EarlyStopping(
                    monitor='val_accuracy',
                    patience=10,
                    restore_best_weights=True,
                    mode='max'
                )

                model_checkpoint = ModelCheckpoint(
                    filepath='best_simple_model.keras',
                    monitor='val_accuracy',
                    save_best_only=True,
                    mode='max'
                )

                reduce_lr = ReduceLROnPlateau(
                    monitor='val_accuracy',
                    factor=0.5,
                    patience=5,
                    min_lr=0.0001,
                    mode='max',
                    verbose=1
                )

                # Train the model
                print("Training simple model...")
                history = model.fit(
                    training_set,
                    steps_per_epoch=steps_per_epoch,
                    epochs=30,  # Reduced epochs for faster iteration
                    validation_data=validation_set,
                    validation_steps=validation_steps,
                    callbacks=[early_stopping, model_checkpoint, reduce_lr],
                    verbose=1
                )

                # Load the best weights
                if os.path.exists('best_simple_model.keras'):
                    try:
                        model = tf.keras.models.load_model('best_simple_model.keras')
                        print("Loaded best model weights from checkpoint.")
                    except Exception as e:
                        print(f"Error loading best model: {e}")

                # Evaluate on test set
                print("Evaluating model on test set...")
                test_loss, test_accuracy = model.evaluate(test_set, steps=test_steps)
                print(f"Test accuracy: {test_accuracy:.4f}")

                # Plot training history
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 2, 1)
                plt.plot(history.history['accuracy'])
                plt.plot(history.history['val_accuracy'])
                plt.title('Model accuracy')
                plt.ylabel('Accuracy')
                plt.xlabel('Epoch')
                plt.legend(['Train', 'Validation'], loc='upper left')

                plt.subplot(1, 2, 2)
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.title('Model loss')
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                plt.legend(['Train', 'Validation'], loc='upper left')
                plt.tight_layout()
                plt.savefig('training_history.png')

                # Save the model
                model.save(model_path)

            except Exception as e:
                print(f"Error during training: {e}")
                import traceback
                traceback.print_exc()

        # Print model information
        print("Model architecture summary:")
        model.summary()

        # Print model size
        if os.path.exists(model_path):
            model_size_bytes = os.path.getsize(model_path)
            model_size_mb = model_size_bytes / (1024 * 1024)
            print(f"Model size: {model_size_mb:.2f} MB")

        return model, training_set.class_indices

    except Exception as e:
        print(f"Error in model initialization: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def predict_image(model, class_indices, image_path, img_size=(64, 64)):
    """Predict image class with simple but effective ensemble approach"""
    try:
        # Load and preprocess the image
        test_image = image.load_img(image_path, target_size=img_size)
        test_image = image.img_to_array(test_image)
        test_image = simple_preprocess(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        # Get predictions with test-time augmentation
        predictions = []

        # Original image
        result = model.predict(test_image, verbose=0)
        predictions.append(result[0][0])

        # Horizontal flip
        flipped = tf.image.flip_left_right(test_image[0])
        flipped = np.expand_dims(flipped, axis=0)
        result = model.predict(flipped, verbose=0)
        predictions.append(result[0][0])

        # Slight zoom
        height, width, channels = test_image[0].shape
        crop_size = int(min(height, width) * 0.9)
        crop_y = (height - crop_size) // 2
        crop_x = (width - crop_size) // 2
        cropped = test_image[0][crop_y:crop_y+crop_size, crop_x:crop_x+crop_size, :]
        cropped = tf.image.resize(cropped, [height, width])
        cropped = np.expand_dims(cropped, axis=0)
        result = model.predict(cropped, verbose=0)
        predictions.append(result[0][0])

        # Calculate average prediction
        raw_score = np.mean(predictions)

        # Map to class
        prediction = 'dog' if raw_score > 0.5 else 'cat'
        confidence = raw_score if raw_score > 0.5 else 1 - raw_score

        # Print detailed prediction info
        print(f"\nPrediction for {image_path}:")
        print(f"Raw score: {raw_score:.4f}")
        print(f"Class: {prediction}")
        print(f"Confidence: {confidence:.2f}")

        # Provide a confidence assessment
        if confidence > 0.90:
            confidence_text = "very high confidence"
        elif confidence > 0.75:
            confidence_text = "high confidence"
        elif confidence > 0.60:
            confidence_text = "moderate confidence"
        else:
            confidence_text = "low confidence"

        print(f"Assessment: {confidence_text}")

        return prediction, confidence, raw_score

    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return None, 0, 0

if __name__ == "__main__":
    setup_gpu()
    model, class_indices = init_simple_model("tiny_model.keras")

    if model is not None:
        # Test predictions
        print("\nTesting model predictions:")
        predict_image(model, class_indices, 'dataset/single_prediction/cat_or_dog_1.jpg')
        predict_image(model, class_indices, 'dataset/single_prediction/cat_or_dog_2.jpg')
    else:
        print("Model initialization failed. Please check the error messages above.")
