import tensorflow as tf
import tf2onnx
import onnx

def convert_keras_to_onnx(keras_model_path, onnx_model_path):
    try:
        model = tf.keras.models.load_model(keras_model_path)
        spec = tf.TensorSpec((1, 128, 128, 3), tf.float32, name="input")
        # Remove the 'output_names' parameter:
        onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=[spec])
        onnx.save(onnx_model, onnx_model_path)
        print(f"Successfully converted {keras_model_path} to {onnx_model_path}")
    except Exception as e:
        print(f"Error converting Keras model: {e}")


# Example usage:
keras_model_file = "my_model.keras" # Replace with your Keras model file path.
onnx_model_file = "my_model.onnx"   # Replace with your desired ONNX model file path.

if __name__ == "__main__":
    convert_keras_to_onnx(keras_model_file, onnx_model_file)
