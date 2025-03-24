import ezkl
import onnx

run_args = ezkl.PyRunArgs()
run_args.logrows = 10 # This will use 2^10 rows
run_args.num_inner_cols = 1 # Default is 2, reducing can save memory

ezkl.gen_settings('my_model.onnx', 'settings.json', run_args)
ezkl.compile_circuit("my_model.onnx", "my_circuit.ezkl", "settings.json")
ezkl.setup('my_circuit.ezkl', 'vk.key', 'pk.key', 'kzg17.srs')
