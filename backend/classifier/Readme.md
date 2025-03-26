# Classifier server

This server is in charge to expose an API rest and validate if an image is a cat or dog

## Local env
```
cd backend/classifier
python3 -m venv venv
source venv/bin/activate
```


# Run it locally

## EZKL directories
We should create the directories where will generated dynamically the ezkl artifacts
```
mkdir -p /tmp/zk/inputs
mkdir -p /tmp/zk/witnesses
mkdir -p /tmp/zk/proofs
```

## EZKL setup artifacts

Run `ezk/run.sh` and move all generated artefacts to `backend/classifier/artefacts`.


## Run
```
cd backend/
PYTHONPATH=$(pwd) python classifier/main.py
```
