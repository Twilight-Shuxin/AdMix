from profiling.models import ResnetClient, VggClient, InceptionClient, UNet3DClient, GANClient, Deeplabv3Client

model_clients = {
    "resnet50": ResnetClient,
    "vgg19": VggClient,
    "inception_v3": InceptionClient,
    # "llm": LLMClient,
    "unet3d": UNet3DClient,
    "GAN": GANClient,
    "deeplabv3": Deeplabv3Client
}