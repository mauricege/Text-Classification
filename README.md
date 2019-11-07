# Text Classification Models
This repository is forked from https://github.com/TobiasLee/Text-Classification. It adds a commandline interface for convenient model training, evaluation and prediction and updates the code to be compatible with tensorflow 2.

## Dependencies
* python >= 3.7
* tensorflow >= 2.0
* scikit-learn
* click
* pandas

## Very Short Rundown
The models work with csv files containing the columns "title", "content" and "class". Training a model works like this:
```bash
python -m cli [modelname] train [training.csv] --model-dir path/to/model/dir
```
For evaluation after training a model and specifying "path/to/model/dir" as model directory:
```bash
python -m cli [modelname] eval [eval.csv] --model-dir path/to/model/dir
```

For prediction after training a model and specifying "path/to/model/dir" as model directory:
```bash
python -m cli [modelname] predict [predict.csv] --model-dir path/to/model/dir
```


