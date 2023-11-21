"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision

class SegmentationNN(pl.LightningModule):

    def __init__(self, num_classes=23, hparams=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        ########################################################################
        # TODO - Train Your Model                                              #
        #######################################################################


        # MobileNet_large = torchvision.models.mobilenet_v3_large(pretrained=True)

        # self.encoder = nn.Sequential(*list(MobileNet_large.children())[0])  #  7*7*960

        # self.decoder = nn.Sequential(nn.Conv2d(960, 240, kernel_size=1),    #7*7 480
        #                             nn.ReLU(),
        #                             nn.ConvTranspose2d(240, 240, kernel_size=4, padding=1, stride=2), #14,14,240
        #                             nn.ReLU(),
        #                             nn.Conv2d(240, 240, kernel_size=1),     #14*14*120
        #                             nn.ReLU(),
        #                             nn.ConvTranspose2d(240, 120, kernel_size=4, padding=1, stride=2), #28,28,64
        #                             nn.ReLU(),
        #                             nn.ConvTranspose2d(120, 64, kernel_size=4, padding=1, stride=2), #56,56,64
        #                             nn.ReLU(),
        #                             nn.ConvTranspose2d(64, 64, kernel_size=4, padding=1, stride=2), #112,112,64
        #                             nn.ReLU(),
        #                             nn.ConvTranspose2d(64, num_classes, kernel_size=4, padding=1, stride=2), #224,224,23
        #                             )






        


        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        x = self.encoder(x)
        x = self.decoder(x)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

    def general_step(self, batch, batch_idx, mode):
        images, targets = batch
        predictions = self.forward(images)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        loss = criterion(predictions, targets)
        return loss

    def general_end(self, outputs, mode):
        avg_loss = torch.stack([x[mode+'_loss'] for x in outputs]).mean()
        return avg_loss

    def training_step(self, batch, batch_idx ):
        train_loss = self.general_step(batch, batch_idx, 'train')
        self.log("loss", train_loss)
        return {'loss': train_loss}

    def validation_step(self, batch, batch_idx):
        val_loss = self.general_step(batch, batch_idx, 'val')
        self.log("val_loss", val_loss)
        return {'val_loss': val_loss}
        
    def validation_epoch_end(self, outputs):
        avg_loss = self.general_end(outputs, 'val')
        self.log('val_avg_loss', avg_loss, logger=True, prog_bar=True)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.decoder.parameters(), lr=self.hparams['learning_rate'])
        return optim




        
class DummySegmentationModel(pl.LightningModule):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())    #y.size = (240,240)   len(y.size())=2
            y_tensor = y.view(*y.size(), -1)   #torch.Size([240, 240, 1])
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)  #torch.Size([240, 240, 23])
            return zeros.scatter(scatter_dim, y_tensor, 1)     #torch.Size([240, 240, 23])

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)  #torch.Size([1, 23, 240, 240])

    def forward(self, x):
        return self.prediction.float()
