import tensorflow as tf
import tf2onnx
import onnx

def convert_keras_to_onnx(keras_model_path, onnx_model_path):
    try:
        model = tf.keras.models.load_model(keras_model_path)

        # Extract input shape from the model
        input_shape = model.input_shape
        height, width, channels = input_shape[1], input_shape[2], input_shape[3]

        # Create tensor spec with batch size of 1 and the extracted dimensions
        spec = tf.TensorSpec((1, height, width, channels), tf.float32, name="input")

        # Convert the model
        onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=[spec])
        onnx.save(onnx_model, onnx_model_path)
        print(f"Successfully converted {keras_model_path} to {onnx_model_path}")
    except Exception as e:
        print(f"Error converting Keras model: {e}")

# Example usage:
keras_model_file = "tiny_model.keras" # Replace with your Keras model file path.
onnx_model_file = "tiny_model.onnx"   # Replace with your desired ONNX model file path.

if __name__ == "__main__":
    convert_keras_to_onnx(keras_model_file, onnx_model_file)
