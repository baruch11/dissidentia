# DissidentIA
![scheme](data/images/dissidentIA.png?raw=true "")



## Description
Detect political dissidents from the answers of the Grand Débat


## Setup instructions 

Create a virtual environment in .venv at the root of the project

```bash
poetry install
```

## Launch application
Given a pretrained model, named nameModel.pkl and saved in the data/models directory, you can launch the application as follow:

```bash
streamlit run dissidentia/application/app.py BertTypeClassifier

# For a demo with a faster interpretability use baselineModel
streamlit run dissidentia/application/app.py baselineModel

```

## Train 
The train could be computed with the following commande line:

```bash
python dissidentia/application/train.py
```

## Annotation
We annotated the dataset through Quantmetry doccano's platform (https://qmdoccano.azurewebsites.net)

To use a training set directly from doccano platform, you need to set the 
2 environments variables *DOCCANO_LOGIN*, *DOCCANO_PASSWORD*.
Then use the *-doccano* option in train.py command line.

```bash
python dissidentia/application/train.py --doccano 
```

## Model 
A scikit-learn wrapper to fine-tuning Bert type model using huggingface Trainer API for dissidentIA detection is proposed and can be used as follow:

```python3

from dissidentia.domain.sklearn_bert_wrapper import BertTypeClassifier

# define model with default parameters
model = BertTypeClassifier(val_dataset=(x_val, y_val)) 

# fine-tuning model
model.fit(x_train, y_train)

# make predictions
y_pred = model.predict(x_test)

# make probabilty predictions
y_pred = model.predict_proba(x_test)

# evaluate model on val_dataset for different metrics
model.evaluate() 

# save model
model.save(save_path)

# load model 
new_model = model.load(save_path)
```



## Tests

```
# unit tests
pytest

# test camembert model fit/predict/save/load methods on few samples
pytest -o python_functions="camembert_*" -s
```


## Authors and acknowledgment
Amir, Moindzé, Charles  

Thanks Benoit Lebreton for the doccano platform
