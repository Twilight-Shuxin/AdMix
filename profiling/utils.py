import os 
from profiling.models import *

model_workers = {
    "resnet50": ResnetWorker,
    "vgg19": VggWorker,
    "inception_v3": InceptionWorker,
    "bert": BertWorker,
    "llm": LLMWorker
}