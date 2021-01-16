from torch.optim import Adam
from copy import deepcopy
from numpy import mean, std
from tqdm import tqdm
import argparse
import random
from utils.data import *
from model.spgat import *
from model.gat import *
from model.gcn import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(model, data):
    model.eval()

    with torch.no_grad():
        output = model(data)

    outputs = {}
    for key in ['train', 'val', 'test']:
        if key == 'train':
            mask = data.train_mask
        elif key == 'val':
            mask = data.val_mask
        else:
            mask = data.test_mask
        loss = F.nll_loss(output[mask], data.labels[mask]).item()
        pred = output[mask].max(dim=1)[1]
        acc = pred.eq(data.labels[mask]).sum().item() / mask.sum().item()

        outputs['{}_loss'.format(key)] = loss
        outputs['{}_acc'.format(key)] = acc

    return outputs

def train(model, epochs, lr, weight_decay, early_stopping, patience, verbose, niter):
    use_loss, use_acc, save_model = True, True, True
    data.to(DEVICE)
    val_acc_list = []
    test_acc_list = []

    for i in range(niter):
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        model.to(DEVICE).reset_parameters()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # for early stopping
        if early_stopping:
            stop_checker = EarlyStopping(patience, verbose, use_loss, use_acc, save_model)

        for epoch in tqdm(range(1, epochs + 1)):
            model.train()
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output[data.train_mask], data.labels[data.train_mask])
            loss.backward()
            optimizer.step()

            evals = evaluate(model, data)

            if verbose:
                print('epoch: {: 4d}'.format(epoch),
                      'train loss: {:.5f}'.format(evals['train_loss']),
                      'train acc: {:.5f}'.format(evals['train_acc']),
                      'val loss: {:.5f}'.format(evals['val_loss']),
                      'val acc: {:.5f}'.format(evals['val_acc']))

            if early_stopping:
                if stop_checker.check(evals, model, epoch):
                    break

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        evals = evaluate(model, data)

        print('num_iters: {: 4d}'.format(i),
              'train loss: {:.5f}'.format(evals['train_loss']),
              'train acc: {:.5f}'.format(evals['train_acc']),
              'val loss: {:.5f}'.format(evals['val_loss']),
              'val acc: {:.5f}'.format(evals['val_acc']))

        val_acc_list.append(evals['val_acc'])
        test_acc_list.append(evals['test_acc'])

    print("mean", mean(test_acc_list))
    print("std", std(test_acc_list))
    return {
        'val_acc': mean(val_acc_list),
        'test_acc': mean(test_acc_list),
        'test_acc_std': std(test_acc_list)
    }

class EarlyStopping:
    def __init__(self, patience, verbose, use_loss, use_acc, save_model):
        assert use_loss or use_acc, 'use loss or (and) acc'
        self.patience = patience
        self.use_loss = use_loss
        self.use_acc = use_acc
        self.save_model = save_model
        self.verbose = verbose
        self.counter = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0
        self.state_dict = None

    def check(self, evals, model, epoch):
        if self.use_loss and self.use_acc:
            # For GAT, based on https://github.com/PetarV-/GAT/blob/master/execute_cora.py
            if evals['val_loss'] <= self.best_val_loss or evals['val_acc'] >= self.best_val_acc:
                if evals['val_loss'] <= self.best_val_loss and evals['val_acc'] >= self.best_val_acc:
                    if self.save_model:
                        self.state_dict = deepcopy(model.state_dict())
                self.best_val_loss = min(self.best_val_loss, evals['val_loss'])
                self.best_val_acc = max(self.best_val_acc, evals['val_acc'])
                self.counter = 0
            else:
                self.counter += 1
        elif self.use_loss:
            if evals['val_loss'] < self.best_val_loss:
                self.best_val_loss = evals['val_loss']
                self.counter = 0
                if self.save_model:
                    self.state_dict = deepcopy(model.state_dict())
            else:
                self.counter += 1
        elif self.use_acc:
            if evals['val_acc'] > self.best_val_acc:
                self.best_val_acc = evals['val_acc']
                self.counter = 0
                if self.save_model:
                    self.state_dict = deepcopy(model.state_dict())
            else:
                self.counter += 1
        stop = False
        if self.counter >= self.patience:
            stop = True
            if self.verbose:
                print("Stop training, epoch:", epoch)
            if self.save_model:
                model.load_state_dict(self.state_dict)
        return stop

if __name__ == '__main__':

    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gat', help='select model')
    parser.add_argument('--data', type=str, default='cora', help='select dataset')
    parser.add_argument('--epochs', type=int, default=100000, help='Number of epochs to train.')
    parser.add_argument('--niter', type=int, default=10, help='iter value for avg.')
    parser.add_argument('--early_stopping', action='store_true', default=False, help='set early_stopping')
    parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
    parser.add_argument('--seed', type=int, default=72, help='Random seed.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
    parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--patience', type=int, default=100, help='Patience')
    parser.add_argument('--verbose', action='store_true', default=False, help='set early_stopping')

    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = load_data(args.data)

    if args.model == 'spgat':
        model = SPGAT(data,
                      nhead=args.nb_heads,
                      nhid=args.hidden,
                      dropout=args.dropout,
                      alpha=args.alpha)

    elif args.model == 'gcn':
        model = GCN(data,
                nhid=args.hidden,
                dropout=args.dropout)

    elif args.model == 'gat':
        model = GAT(data,
                    nhead=args.nb_heads,
                    nhid=args.hidden,
                    dropout=args.dropout,
                    alpha=args.alpha)
    else:
        model = GAT(data,
                    nhead=args.nb_heads,
                    nhid=args.hidden,
                    dropout=args.dropout,
                    alpha=args.alpha)

    train(model=model,
          epochs=args.epochs,
          lr=args.lr,
          weight_decay=args.weight_decay,
          early_stopping=args.early_stopping,
          patience=args.patience,
          verbose=args.verbose,
          niter=args.niter)

