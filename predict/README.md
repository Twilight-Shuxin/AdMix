# Predict Module

This folder contains files necessary for training the estimator used for predictive scheduler. 

## Notes

Two environment variables could be set in `.env`:

```
OUTPUT_PATH=
MODEL_PATH=
```

`OUTPUT_PATH` is the directory to store all output files such as pre-processed train / test data, plot graphs.
`MODEL_PATH` is the directory to load / store model & scaler. 
Both are set to `outputs` by default.

## Usage

### Prepare data

Provide data file to `prep_data.py` to generate '.pt' files storing train/validation/test data.
If `True` is provided as second cmd arg, entire data file is processed as test data.
```
python prep_data.py data.csv
python prep_data.py data.csv True
```

### Train model

Train a model of `$model_type` to predict `$predict_type` throughput (Default to both prefill & decode).
Training data uses '.pt' data under `OUTPUT_PATH`.
If `MODEL_PATH` is provided and the model exists, training continues from the loaded model and overwrites it, while the scaler is not updated. 
Otherwise, a new model is trained and the scaler is saved.

```
python train_model.py $model_type $predict_type
python train_model.py MLP
python train_model.py MLP prefill
```

### Evaluate model

Evaluate model under `MODEL_PATH` using test `.pt` data under `OUTPUT_PATH`.

```
python eval_model.py $model_type
python eval_model.py MLP
```