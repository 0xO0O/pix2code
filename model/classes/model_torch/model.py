from torch import cat
from torch.nn import \
    Conv2d, Dropout, Flatten, Linear, LSTM, \
    MaxPool2d, Module, ReLU, Sequential, Softmax

from .util import ToTensor

"""
@TODO:
    1. Read about "Liear"
    2. Calculate dimensions
    3. Use data from original paper
    4. Resolve all "TODOs"
    5. Wrapper class for model with "fit" and "predict" methods (fit == training loop)
"""

class ImageEncoder(Module):
    def __init__(self):
        super().__init__()

        self.model = Sequential(
            # Block #1
            Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            ReLU(),
            Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            ReLU(),
            MaxPool2d(kernel_size=2),
            Dropout(p=0.25),

            # Block #2
            Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            ReLU(),
            MaxPool2d(kernel_size=2),
            Dropout(p=0.25),

            # Block #3
            Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=3),
            ReLU(),
            MaxPool2d(kernel_size=2),
            Dropout(p=0.25),

            # Block #4
            Flatten(), # What does flatter do?
            Linear(in_features=128*28*28, out_features=1024),
            ReLU(),
            Dropout(p=0.3),
            Linear(in_features=1024, out_features=1024),
            Dropout(p=0.3),
        )

    def forward(self, x):
        return self.model(x)


class ContextEncoder(Module):
    def __init__(self, contextLength, outputSize):
        super().__init__()
        self.model = LSTM(input_size=contextLength, hidden_size=128, num_layers=2, batch_first=True) 

    def forward(self, x):
        res, _ = self.model(x)
        return res


class Decoder(Module):
    def __init__(self, contextLength, outputSize):
        super().__init__()

        self.model = Sequential(
            # @TODO: calculate input size somehow
            LSTM(input_size=1152, hidden_size=512, num_layers=2, batch_first=True),
            ToTensor(),
            Linear(in_features=512, out_features=outputSize),
            Softmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)


class Pix2Code(Module):
    def __init__(self, contextLength, outputSize):
        super().__init__()
        
        self.contextLength = contextLength
        self.outputSize = outputSize

        self.imageEncoder = ImageEncoder()
        self.contextEncoder = ContextEncoder(contextLength, outputSize)
        self.decoder = Decoder(contextLength, outputSize)        

    def forward(self, image, context):
        encodedImage = self.imageEncoder(image)
        encodedContext = self.contextEncoder(context)
        encodedImage = encodedImage.unsqueeze(1).repeat(1, encodedContext.size(1), 1)

        return self.decoder(cat([encodedImage, encodedContext], dim=2))