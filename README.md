# zkml-catordog

## Instructions
To run the model:
- Create a virtual environment: `python3 -m venv venv && ./venv/bin/activate`
- Install needed components: `pip install tensorflow scipy pillow`
- run `python3 classifier.py`

To train the model from scratch, just delete the `my_model.keras` file and run `python3 classifier.py` again.

To convert the model from `.keras` to `.onnx`, run `python convert.py`