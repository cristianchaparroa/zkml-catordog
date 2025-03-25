#!/usr/bin/env bash

# setup
ezkl gen-srs --srs-path srs17.srs --logrows 17
ezkl gen-settings -M ../classifier/tiny_model.onnx -K 17 -N 1 --scale-rebase-multiplier 2
ezkl compile-circuit -M ../classifier/tiny_model.onnx -S settings.json --compiled-circuit my_circuit.ezkl
ezkl setup -M my_circuit.ezkl --srs-path srs17.srs --vk-path vk.key --pk-path pk.key

# prove
ezkl gen-random-data -M ../classifier/tiny_model.onnx --data input.json
ezkl gen-witness --compiled-circuit my_circuit.ezkl --data input.json --output witness.json
ezkl prove --witness witness.json --compiled-circuit my_circuit.ezkl --pk-path pk.key --srs-path srs17.srs --proof-path proof.json
