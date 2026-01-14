from profiling.models import ResnetWorker, VggWorker, InceptionWorker, LLMWorker, UNet3DWorker, GANWorker, Deeplabv3Worker

model_workers = {
    "resnet50": ResnetWorker,
    "vgg19": VggWorker,
    "inception_v3": InceptionWorker,
    "llm": LLMWorker,
    "unet3d": UNet3DWorker,
    "GAN": GANWorker,
    "deeplabv3": Deeplabv3Worker
}