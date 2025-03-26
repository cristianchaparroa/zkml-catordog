#!/usr/bin/env bash

# Define variables with original paths
MODEL_PATH=../classifier/tiny_model.onnx
CIRCUIT_PATH=my_circuit.ezkl
VK_PATH=vk.key
PK_PATH=pk.key
SRS_PATH=srs17.srs
SETTINGS_PATH=settings.json
WITNESS_PATH=witness.json
PROOF_PATH=proof.json
INPUT_PATH=input.json

# setup (developer)
ezkl gen-srs --srs-path $SRS_PATH --logrows 17
ezkl gen-settings -M $MODEL_PATH -K 17 -N 1 --scale-rebase-multiplier 2
ezkl compile-circuit -M $MODEL_PATH -S $SETTINGS_PATH --compiled-circuit $CIRCUIT_PATH
ezkl setup -M $CIRCUIT_PATH --srs-path $SRS_PATH --vk-path $VK_PATH --pk-path $PK_PATH

# prove (user interaction)
# input simulates the user input and we will generate a proof around it
# ezkl gen-random-data -M $MODEL_PATH --data $INPUT_PATH
# ezkl gen-witness --compiled-circuit $CIRCUIT_PATH --data $INPUT_PATH --output $WITNESS_PATH
# the proof will be passed to the verifier to validate if it is valid
# ezkl prove --witness $WITNESS_PATH --compiled-circuit $CIRCUIT_PATH --pk-path $PK_PATH --srs-path $SRS_PATH --proof-path $PROOF_PATH

# verify
# the verifier takes the proof and validate the proof
# ezkl verify --srs-path $SRS_PATH --vk-path $VK_PATH --proof-path $PROOF_PATH --settings-path $SETTINGS_PATH
