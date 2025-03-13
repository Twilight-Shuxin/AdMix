import os 
from profiling.models import *

model_worker = {
    "resnet50": ResnetWorker,
    "vgg19": VggWorker,
    "inception_v3": InceptionWorker,
    "bert": BertWorker,
    "llm": LLMWorker
}