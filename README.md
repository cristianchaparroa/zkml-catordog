# zkml-catordog

The following folder structure is suggested for the different artifacts we will generate for this project

```
classifiers/
circuits/
ezkl/
backend/
fronted/
```


## Instructions
To run the model:
- Create a virtual environment: `python3 -m venv venve && ./venv/bin/activate`
- Install needed components: `pip install tensorflow scipy pillow`
- run `python3 classifier.py`

To train the model from scratch, just delete the `my_model.keras` file and run `python3 classifier.py` again.