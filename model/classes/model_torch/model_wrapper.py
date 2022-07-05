from torch import Tensor, argmax, save
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torch.optim import RMSprop
from torch.utils.data import DataLoader, TensorDataset
from .model import Pix2Code
from ..model.Config import BATCH_SIZE, CONTEXT_LENGTH, EPOCHS


class Pix2CodeWrapper(Pix2Code):
    def __init__(self, input_shape, output_size, output_path):
        super().__init__(CONTEXT_LENGTH, output_size)
        self.outputPath = output_path

    def fit(self, images, contexts, predictions, dataset = None):
        loader = DataLoader(
            dataset = TensorDataset(Tensor(images), Tensor(contexts), Tensor(predictions)),
            batch_size = BATCH_SIZE,
            shuffle = False,
        )
        criterion = CrossEntropyLoss()
        optimizer = RMSprop(self.parameters(), lr = 0.0001)

        for epoch in range(EPOCHS):
            self.zero_grad()

            for step, (image, context, prediction) in enumerate(loader):
                output = self(image.permute(0, 3, 1, 2), context.permute(0, 2, 1))
                output = argmax(output, 1)
                prediction = argmax(prediction, 1)

                loss = criterion(output.float(), prediction)
                loss = Variable(loss, requires_grad = True)
                loss.backward()

                if step%10 == 0:
                    optimizer.step()
                    print('Loss: {}, Epoch: {}'.format(loss.data, epoch))
                    self.zero_grad()

        self.save()

    def fit_generator(self, generator, steps_per_epoch):
        criterion = CrossEntropyLoss()
        optimizer = RMSprop(self.parameters(), lr = 0.0001)

        for epoch in range(EPOCHS):
            self.zero_grad()

            for step, ([image, context], prediction) in enumerate(generator):
                output = self(Tensor(image).permute(0, 3, 1, 2), Tensor(context).permute(0, 2, 1))
                output = argmax(output, 1)
                prediction = argmax(Tensor(prediction), 1)

                loss = criterion(output.float(), prediction)
                loss = Variable(loss, requires_grad = True)
                loss.backward()

                if step%10 == 0:
                    optimizer.step()
                    print('Loss: {}, Epoch: {}'.format(loss.data, epoch))
                    self.zero_grad()

        self.save()

    def save(self):
        save(self.state_dict(), self.outputPath)

    def predict(self):
        raise Exception("Method is not implemented!")

    def predict_batch(self, _1, _2):
        raise Exception("Method is not implemented!")

    def load_weights(self, _1):
        raise Exception("Method is not implemented!")