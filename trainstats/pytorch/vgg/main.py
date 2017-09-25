import torch
import torchvision.models as models
from matplotlib import pyplot as plt

def main():
    vgg = models.vgg16(pretrained=True)
    mm = vgg.double()
    body_model = [i for i in mm.children()][0]
    layer1 = body_model[0]
    tensor = layer1.weight.data.numpy()

if __name__=="__main__":
    main()
