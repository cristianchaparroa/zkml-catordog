import ezkl
import onnx

ezkl.gen_settings('my_model.onnx')
ezkl.compile_circuit("my_model.onnx", "my_circuit.ezkl", "settings.json")
ezkl.setup('my_circuit.ezkl', 'vk.key', 'pk.key', 'kzg17.srs')
