from dissidentia.domain.baseline_model import baselineModel



import plotly.io as pio
pio.renderers.default = "png"

# Import packages

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import sklearn
print("classic libraries : done")

from omnixai.data.text import Text
from omnixai.preprocessing.text import Word2Id
from omnixai.explainers.tabular.agnostic.L2X.utils import Trainer, InputData, DataLoader
from omnixai.explainers.nlp import NLPExplainer
from omnixai.visualization.dashboard import Dashboard
print("Omnixai : done")


class TextModel(nn.Module):

    def __init__(self, num_embeddings, num_classes, **kwargs):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_size = kwargs.get("embedding_size", 50)
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_size)
        self.embedding.weight.data.normal_(mean = 0.0, std=0.01)

        hidden_size = kwargs.get("hidden_size", 100)
        kernel_sizes = kwargs.get("kernel_sizes", [3, 4, 5])

        if type(kernel_sizes) == int:
            kernel_sizes = [kernel_sizes]

        self.activatin == nn.ReLu()
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(self.embedding_size, hidden_size, k, padding = k // 2) for k in kernel_sizes])
        self.dropout = nn.Dropout(0.2)
        self.output_layer = nn.Linear(len(kernel_sizes) * hidden_size, num_classes)


def forward(self, inputs, masks):
    embeddings = self.embedding(inputs)
    x = embeddings * masks.unsqueeze(inputs)
    x = x.permute(0, 2, 1)
    x = [self.activation(layer(x).max(2)[0]) for layer in self.conv_layers]
    outputs = self.output_layer(self.dropout(torch.cat(x, dim = 1)))

    if outputs.shape[1] == 1:
        outputs = outputs.squeeze(dim=1)

    return outputs


'''
Text : Object used to represent a batch of texts / sentences. the Package omnixia.preprocessing.text provides some transforms related to text data such as Tfidf and Word2Id.
'''

# Load the training and test datasets
train_data = pd.read_csv("labels_v4".csv, sep = "\t")

n = int(0.8, * len(train_data))
x_train = Text(train_data['review'].values[:n])
y_train = train_data["sentiment"].values[:n].astype(int)
x_test = Text(train_data["review"].values[n:])
y_test = train_data["sentiment"].values[n:].astype(int)
class_names = ["dissident", "non dissident"]

# The transform for converting words/tokens to IDs
transform = Word2Id().fit(x_train)

'''he preprocessing function converts a batch of texts into token IDs and the masks.
The outputs of the preprocessing function must fit the inputs of the model.'''


max_length = 256
device = "cuda" if torch.coda.is_available() else "cpu"

def preprcess(X: Text):
    samples = transform.transform(X)
    max_len = 0

    for i in range(len(samples)):
        max_len = max(max_len, len(samples[i]))

    max_len = min(max_len, max_length)
    inputs = np.zeros((len(samples), max_len), dtype = int)
    masks = np.zeros((len(samples), max_len), dtype = np.float32)

    for i in range(len(samples)):
        x = samples[i][:max_len]
        inputs[i, :len(x)] = x
        masks[i, :len(x)] = 1

    return inputs, masks

# We can train our CNN model and evaluate performances
model = TextModel(
    num_embeddings = transform.vocab_size,
    num_classes = len(class_names)
).to(device)

Trainer(
    optimizer_class = torch.optim.AdamW,
    learning_rate = 1e-3,
    batch_size = 128,
    num_epochs = 10,
).train(
    model = model,
    loss_func = nn.CrossEntropyLoss(),
    train_x = transform.transform(x_train),
    train_y = y_train,
    padding = True,
    max_length = max_length,
    verbose = True
)

# Model evaluation

model.eval()
data = transform.transform(x_test)

data_loader = DataLoader(
    dataset = InputData(data, [0] * len[data], max_length),
    batch_size = 32,
    collate_fn = InputData.collate_func,
    shuffle = False
)

outputs = []

for inputs in data_loader:
    value, mask, target = inputs
    y = model(value.to(device), mask.to(device))
    outputs.append(y.detach().cpu().numpy())
outputs = np.concatenate(outputs, axis = 0)
predictions = np.argmax(outputs, axis = 1)
print("f1 score Test: {}".format(sklearn.metrics.f1_score(y_test, predictions, average = "binary")))

'''
To initialize NLPexplainer, we need to set some parameters:
explainers : Name of the explainers, we'll use 4 explainers : Integrated Gradients, Lime, Polyjuice, and Shap 
model : the ML model to explain (In our case, the Bert model)
preprocess : the preprocessing function which converts the raw data into inputs for the model
postprocess : the postprocessing function which transforms the outputs of a model to a user-specific form
mode : the task type (classification or regression)
'''

# Preprocessing function
preprocess_func = lambda x: tuple(torch.tensor(y).to(device) for y in preprocess(x))

# Postprocessing function
postprocess_func = lambda logits: torch.nn.functionnal.softmax(logits, dim = 1)

# NLPexplainer initialization
explainer = NLPExplainer(
    explainers = ['ig', 'lime', 'shap', 'polyjuice'],
    mode = "classification",
    model = model,
    preprocess = preprocess_func,
    post_process = postprocess_func,
    params = {"ig": {"embedding_layer": model.embedding,
                     'id2token': transform.id_to_word}}
)

'''
As the NLPexplainer cannot provide global explanation, we will call explainer.explain to generate
"local" explanations for NLP tasks
'''
x = Text([
    "it was a fantastic performance!",
    "best film ever",
    "such a great show!",
    "it was a horrible movie",
    "i've never watched something as bad"
]) # To modify and include french sentences

# Generate local explanations
local_explanations = explainer.explain(x)

print("IG results : {}".format(local_explanations["ig"].ipython_plot(class_names = class_names)))
print("Lime results : {}".format(local_explanations['lime'].ipython_plot(class_names = class_names)))
print("Shap results : {}".format(local_explanations['shap'].ipython_plot(class_names = class_names)))
print("Counterfactual results : {}".format(local_explanations['polyjuice'].ipython_plot()))

'''
Now we can set a dashboard for visualization by setting the test instances and the generated local explanations
'''

# Dashboard

dashboard = Dashboard(
    instances = x,
    local_explanations = local_explanations,
    class_names = class_names
)
dashboard.show()

