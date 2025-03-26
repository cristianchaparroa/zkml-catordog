# Catordog classifier

## Local environment

```
cd classifier
```

- Create a virtual environment:
```
python3 -m venv venv
source venv/bin/activate

```
- Install needed components: 
```
pip install -r requirements.txt
```

## Run the model

Running the model for inference
```
python3 classifier.py
```

## Re-training the model
To train the model from scratch, just delete the `my_model.keras` file and run `python3 classifier.py` again.


## Convert Keras model to ONNX
```
python3 converter.py
```

