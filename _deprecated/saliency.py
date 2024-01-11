import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from math import ceil
from time import time

from ketos.neural_networks.resnet import ResNetInterface

import xplique
from xplique.attributions import Saliency, DeconvNet, GradientInput, GuidedBackprop
from xplique.plots import plot_attributions


# The model file needs to be a keras "Model" object
model_file = r'C:\Users\kzammit\Documents\Detector\detector-1sec\rs-1sec-3.kt'

resnet = ResNetInterface.load(model_file=model_file)

sample_data = np.ones([1, 1500, 56, 1])

resnet.model(sample_data)

parameters = {
    "model": resnet.model,
    "output_layer": None,
    "batch_size": 16,
}

# instanciate one explainer for each method
explainers = {
    "Saliency": Saliency(**parameters),
    "DeconvNet": DeconvNet(**parameters),
    "GradientInput": GradientInput(**parameters),
    "GuidedBackprop": GuidedBackprop(**parameters),
}

# iterate on all methods
for method_name, explainer in explainers.items():
    # compute explanation by calling the explainer
    explanation = explainer.explain(x, y)

    # visualize explanation with plot_explanation() function
    print(method_name)
    plot_attributions(explanation, x, img_size=5, cmap='cividis', cols=1, alpha=0.6)
    plt.show()
