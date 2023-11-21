from unicodedata import bidirectional
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

from .rnn_nn import Embedding, RNN, LSTM
import pytorch_lightning as pl
import torch



class RNNClassifier(pl.LightningModule):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, use_lstm=True, **additional_kwargs):
        """
        Inputs:
            num_embeddings: size of the vocabulary
            embedding_dim: size of an embedding vector
            hidden_size: hidden_size of the rnn layer
            use_lstm: use LSTM if True, vanilla RNN if false, default=True
        """
        super().__init__()

        # Change this if you edit arguments
        hparams = {
            'num_embeddings': num_embeddings,
            'embedding_dim': embedding_dim,
            'hidden_size': hidden_size,
            'use_lstm': use_lstm,
            **additional_kwargs
        }
        # if you do not inherit from lightning module use the following line
        # self.hparams = hparams
        
        # if you inherit from lightning module, comment out the previous line and use the following line
        self.hparams.update(hparams)
        
        ########################################################################
        # TODO: Initialize an RNN network for sentiment classification         #
        # hint: A basic architecture can have an embedding, an rnn             #
        # and an output layer                                                  #
        ########################################################################
        

        self.Embedding = nn.Embedding(hparams['num_embeddings'],hparams['embedding_dim'], padding_idx=0)
        if use_lstm == True:
            self.rnn = nn.LSTM(hparams['embedding_dim'], hparams['hidden_size'], dropout=0.5, num_layers=1)

        else:
            self.rnn = nn.RNN(hparams['embedding_dim'],hparams['hidden_size'])

        self.fc = nn.Sequential(nn.Linear(hparams['hidden_size'], 128),
                                nn.Linear(128, 64),
                                nn.Tanh(),
                                nn.Linear(64, 1),
                                nn.Sigmoid())

        
        


        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, sequence, lengths=None):
        """
        Inputs
            sequence: A long tensor of size (seq_len, batch_size)
            lengths: A long tensor of size batch_size, represents the actual
                sequence length of each element in the batch. If None, sequence
                lengths are identical.
        Outputs:
            output: A 1-D tensor of size (batch_size,) represents the probabilities of being
                positive, i.e. in range (0, 1)
        """
        output = None

        ########################################################################
        # TODO: Apply the forward pass of your network                         #
        # hint: Don't forget to use pack_padded_sequence if lenghts is not None#
        # pack_padded_sequence should be applied to the embedding outputs      #
        ########################################################################
        x = self.Embedding(sequence)
        if lengths is not None:
            x = pack_padded_sequence(input=x, lengths=lengths)

        
        _, (hn, _) = self.rnn(x)  #output.size : (seqLen, batchsize, hiddensize)

        output = hn[-1]
        output = self.fc(output)     #(batchsize, )
        output = torch.squeeze(output)

        

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return output

    def general_step(self, batch, batch_idx, mode):
        text, label = batch['data'], batch['label']
        predictions = self.forward(text)
        criterion = nn.BCELoss()
        loss = criterion(predictions, label)
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
        optim = torch.optim.Adam(self.parameters(), lr=0.001,weight_decay=1e-3)
        return optim