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

## Authors and acknowledgment
Amir, Moindzé, Charles  

Thanks Benoit Lebreton for the doccano platform
