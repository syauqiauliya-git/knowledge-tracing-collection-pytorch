import os

import numpy as np
import torch

from torch.nn import Module, Embedding, LSTM, Linear, Dropout
from torch.nn.functional import one_hot, binary_cross_entropy
from sklearn import metrics


class DKT(Module):
    '''
        Args:
            num_q: the total number of the questions(KCs) in the given dataset
            emb_size: the dimension of the embedding vectors in this model
            hidden_size: the dimension of the hidden vectors in this model
    '''
    def __init__(self, num_q, emb_size, hidden_size, dropout_rate=0.5):
        super().__init__()
        self.num_q = num_q
        self.emb_size = emb_size
        self.hidden_size = hidden_size

        self.interaction_emb = Embedding(self.num_q * 2, self.emb_size)
        self.lstm_layer = LSTM(
            self.emb_size, self.hidden_size, batch_first=True
        )
        self.out_layer = Linear(self.hidden_size, self.num_q)
        self.dropout_layer = Dropout(p=dropout_rate)


    def forward(self, q, r):
        '''
            Args:
                q: the question(KC) sequence with the size of [batch_size, n]
                r: the response sequence with the size of [batch_size, n]

            Returns:
                y: the knowledge level about the all questions(KCs)
        '''
        x = q + self.num_q * r
        h, _ = self.lstm_layer(self.interaction_emb(x))
        h = self.dropout_layer(h)
        y = self.out_layer(h)
        y = torch.sigmoid(y)
        return y

    def train_model(
        self, train_loader, test_loader, num_epochs, opt, ckpt_path,
        early_stopping_patience=3, early_stopping_delta=0.0
    ):
        '''
            Args:
                train_loader: the PyTorch DataLoader instance for training
                test_loader: the PyTorch DataLoader instance for test
                num_epochs: the number of epochs
                opt: the optimization to train this model
                ckpt_path: the path to save this model's parameters
                early_stopping_patience: number of epochs to wait before early stop
                early_stopping_delta: minimum change to qualify as improvement
        '''
        aucs = []
        loss_means = []
        max_auc = 0
        best_loss = None
        patience_counter = 0

        for i in range(1, num_epochs + 1):
            self.train()
            epoch_losses = []

            for data in train_loader:
                q, r, qshft, rshft, m = data

                y = self(q.long(), r.long())
                y = (y * one_hot(qshft.long(), self.num_q)).sum(-1)

                y = torch.masked_select(y, m)
                t = torch.masked_select(rshft, m)

                opt.zero_grad()
                loss = binary_cross_entropy(y, t)
                loss.backward()
                opt.step()

                epoch_losses.append(loss.item())

            loss_mean_epoch = np.mean(epoch_losses)

            self.eval()
            with torch.no_grad():
                y_true_all, y_pred_all = [], []

                for data in test_loader:
                    q, r, qshft, rshft, m = data

                    y = self(q.long(), r.long())
                    y = (y * one_hot(qshft.long(), self.num_q)).sum(-1)

                    y = torch.masked_select(y, m).cpu().numpy()
                    t = torch.masked_select(rshft, m).cpu().numpy()

                    y_true_all.extend(t.astype(int))
                    y_pred_all.extend(y)

                auc = metrics.roc_auc_score(y_true_all, y_pred_all)
                print(
                    "Epoch: {},   AUC: {:.4f},   Loss Mean: {:.4f}".format(i, auc, loss_mean_epoch)
                )

                if auc > max_auc:
                    torch.save(
                        self.state_dict(),
                        os.path.join(ckpt_path, "model.ckpt")
                    )
                    max_auc = auc

                aucs.append(auc)
                loss_means.append(loss_mean_epoch)

            # Early stopping check
            if best_loss is None:
                best_loss = loss_mean_epoch
            elif loss_mean_epoch > best_loss - early_stopping_delta:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"⏹️ Early stopping at epoch {i} due to no improvement")
                    break
            else:
                best_loss = loss_mean_epoch
                patience_counter = 0

        return aucs, loss_means
