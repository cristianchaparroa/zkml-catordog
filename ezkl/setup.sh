#!/usr/bin/env bash

ezkl gen-srs --srs-path srs7.srs --logrows 7
ezkl gen-settings -M ../classifier/tiny_model.onnx -K 7 -N 1
ezkl compile-circuit -M ../classifier/tiny_model.onnx -S settings.json --compiled-circuit my_circuit.ezkl
ezkl setup -M my_circuit.ezkl --srs-path srs7.srs --vk-path vk.key --pk-path pk.key
