"""Models for facial keypoint detection"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision

class Inception(nn.Module):
    def __init__(self, in_channels):
        super(Inception,self).__init__()
        self.branch1 = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                                    nn.Conv2d(in_channels, 24, kernel_size=1))
        self.branch2 = nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch3 = nn.Sequential(nn.Conv2d(in_channels, 16, kernel_size=1),
                                    nn.Conv2d(16, 24, kernel_size=5, padding=2))
        self.branch4 = nn.Sequential(nn.Conv2d(in_channels, 16, kernel_size=1),
                                    nn.Conv2d(16, 24, kernel_size=3, padding=1),
                                    nn.Conv2d(24, 24, kernel_size=3, padding=1))

    def forward(self, x):
        x_branch1 = self.branch1(x)
        x_branch2 = self.branch2(x)
        x_branch3 = self.branch3(x)
        x_branch4 = self.branch4(x)
        outputs = [x_branch1, x_branch2, x_branch3, x_branch4]
        return torch.cat(outputs, dim=1)


# TODO: Choose from either model and uncomment that line
# class KeypointModel(nn.Module):
class KeypointModel(pl.LightningModule):
    """Facial keypoint detection model"""
    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
            
        NOTE: You could either choose between pytorch or pytorch lightning, 
            by switching the class name line.
        """
        super().__init__()
        self.save_hyperparameters(hparams)
        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        #                                                                      #
        # We would truly recommend to make your code generic, such as you      #
        # automate the calculation of the number of parameters at each layer.  #
        # You're going probably try different architecutres, and that will     #
        # allow you to be quick and flexible.                                  #
        ########################################################################
        

        pretrained_M_v3_large = torchvision.models.mobilenet_v3_large(pretrained=True)

        self.conv_layer = nn.Sequential(*list(pretrained_M_v3_large.children())[:-1]) 
        

        self.dense_layer = nn.Sequential(nn.Linear(960, 576),
                                        nn.Tanh(),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(576, 256),
                                        nn.Tanh(),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(256, 30),)
     
        

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        
        # check dimensions to use show_keypoint_predictions later
        if x.dim() == 3:
            x = torch.unsqueeze(x, 0)
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints.                                   #
        # NOTE: what is the required output size?                              #
        ########################################################################
    
        x = self.conv_layer(x)
        x = self.dense_layer(x.view(x.shape[0], -1))
        
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return x

    def general_step(self, batch, batch_idx, mode):
        images, keypoints = batch['image'], batch['keypoints']   # torch.Size([batch_size, 1, 96, 96]) torch.Size([batch_size, 15, 2])
        predicted_keypoints = self.forward(images).view(-1,15,2)
        criterion = torch.nn.MSELoss()
        loss = criterion(torch.squeeze(keypoints), torch.squeeze(predicted_keypoints))
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
        optim = torch.optim.Adam(self.dense_layer, lr=self.hparams['learning_rate'])
        return optim

        

        


class DummyKeypointModel(pl.LightningModule):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1) #x.size = 1,96, 96
