# DissidentIA
![scheme](data/images/dissidentIA.png?raw=true "")



## Description
Detect political dissidents from the answers of the Grand Débat


## Annotation

We annotated the dataset through Quantmetry doccano's platform (https://qmdoccano.azurewebsites.net)

To use a training set directly from doccano platform, you need to set the 
2 environments variables *DOCCANO_LOGIN*, *DOCCANO_PASSWORD*.
Then use the *-doccano* option in train.py command line.

`python dissidentia/application/train.py --doccano`

## Model 

A scikit-learn wrapper to fine-tuning Bert type model using huggingface Trainer API for dissidentIA detection is proposed and can be used as follow:

```python3

from dissidentia.domain.sklearn_bert_wrapper import BertTypeClassifier

# define model with default parameters
model = BertTypeClassifier() 

# fine-tuning model
model.fit(x_train, y_train)

# make predictions
y_pred = model.predict(x_test)

# make probabilty predictions
y_pred = model.predict_proba(x_test)
```
## Application

Given a pretrained model, named nameModel.pkl and saved in the data/models directory, you can run the application as follow:

`run dissidentia/application/app.py -- namedModel` or 

`run dissidentia/application/app.py ` to use the default model


## tests

Run packaging tests with pytest as follows:
```bash
python -m pytest -s tests/
```

## Authors and acknowledgment
Amir, Moindzé, Charles  

Thanks Benoit Lebreton for the doccano platform
