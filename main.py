import numpy as np
import pandas as pd
from modules import FTTransformer, MLP, ResNet, FDS
from lds import LDS
import scipy.special
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import zero

import seaborn as sns
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# CPU only
parser.add_argument('--cpu_only', action='store_true', default=False, help='whether to use CPU only')
# Data
parser.add_argument('--data_dir', type=str, default='D:/UserData/AndyCCChiang/Boston_Housing_dataset/data/processdata.csv', help='data directory')
# LDS
parser.add_argument('--lds', action='store_true', default=False, help='whether to enable LDS')
parser.add_argument('--lds_kernel', type=str, default='gaussian',
                    choices=['gaussian', 'triang', 'laplace'], help='LDS kernel type')
parser.add_argument('--lds_ks', type=int, default=9, help='LDS kernel size: should be odd number')
parser.add_argument('--lds_sigma', type=float, default=1, help='LDS gaussian/laplace kernel sigma')
# FDS
parser.add_argument('--fds', action='store_true', default=False, help='whether to enable FDS')
parser.add_argument('--fds_kernel', type=str, default='gaussian',
                    choices=['gaussian', 'triang', 'laplace'], help='FDS kernel type')
parser.add_argument('--fds_ks', type=int, default=5, help='FDS kernel size: should be odd number')
parser.add_argument('--fds_sigma', type=float, default=1, help='FDS gaussian/laplace kernel sigma')
parser.add_argument('--start_update', type=int, default=0, help='which epoch to start FDS updating')
parser.add_argument('--start_smooth', type=int, default=1, help='which epoch to start using FDS to smooth features')
parser.add_argument('--bucket_num', type=int, default=100, help='maximum bucket considered for FDS')
parser.add_argument('--bucket_start', type=int, default=0, choices=[0, 3],
                    help='minimum(starting) bucket for FDS, 0 for IMDBWIKI, 3 for AgeDB')
parser.add_argument('--fds_mmt', type=float, default=0.9, help='FDS momentum')
# re-weighting: SQRT_INV / INV
parser.add_argument('--reweight', type=str, default='none', choices=['none', 'sqrt_inv', 'inverse'], help='cost-sensitive reweighting scheme')
# FT-Transformer
parser.add_argument('--task_type', type=str, default='regression', choices=['binclass', 'multiclass', 'regression'], help='choose the task type')
parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='optimizer weight decay')
parser.add_argument('--cat_cardinalities', type=float, default=None, help='cat_cardinalities')
parser.add_argument('--epoch', type=int, default=10, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')

parser.set_defaults(augment=True)
args, unknown = parser.parse_known_args()
#%%
if args.cpu_only:
    device = torch.device('cpu')
else:
    device = torch.device('gpu')

# device = torch.device('cpu')
#%%
task_type = args.task_type
# task_type = 'regression'
assert task_type in ['binclass', 'multiclass', 'regression']
# data = pd.read_csv("D:/UserData/AndyCCChiang/Boston_Housing_dataset/data/processdata.csv", encoding='big5', low_memory=False)
# x_train = pd.read_csv("D:/UserData/AndyCCChiang/Boston_Housing_dataset/data/x_train.csv", header=None).astype('float32')
# y_train = np.array(pd.read_csv("D:/UserData/AndyCCChiang/Boston_Housing_dataset/data/y_train.csv", header=None)).flatten().astype('float32' if task_type == 'regression' else 'int64')
# x_val = pd.read_csv("D:/UserData/AndyCCChiang/Boston_Housing_dataset/data/x_val.csv", header=None).astype('float32')
# y_val = np.array(pd.read_csv("D:/UserData/AndyCCChiang/Boston_Housing_dataset/data/y_val.csv", header=None)).flatten().astype('float32' if task_type == 'regression' else 'int64')
# x_test = pd.read_csv("D:/UserData/AndyCCChiang/Boston_Housing_dataset/data/x_test.csv", header=None).astype('float32')
# y_test = np.array(pd.read_csv("D:/UserData/AndyCCChiang/Boston_Housing_dataset/data/y_test.csv", header=None)).flatten().astype('float32' if task_type == 'regression' else 'int64')
data = pd.read_csv(args.data_dir, encoding='big5', low_memory=False)
x_train = pd.read_csv("D:/UserData/AndyCCChiang/Boston_Housing_dataset/data/x_train.csv", header=None).astype('float32')
y_train = np.array(pd.read_csv("D:/UserData/AndyCCChiang/Boston_Housing_dataset/data/y_train.csv", header=None)).flatten().astype('float32' if task_type == 'regression' else 'int64')
x_val = pd.read_csv("D:/UserData/AndyCCChiang/Boston_Housing_dataset/data/x_val.csv", header=None).astype('float32')
y_val = np.array(pd.read_csv("D:/UserData/AndyCCChiang/Boston_Housing_dataset/data/y_val.csv", header=None)).flatten().astype('float32' if task_type == 'regression' else 'int64')
x_test = pd.read_csv("D:/UserData/AndyCCChiang/Boston_Housing_dataset/data/x_test.csv", header=None).astype('float32')
y_test = np.array(pd.read_csv("D:/UserData/AndyCCChiang/Boston_Housing_dataset/data/y_test.csv", header=None)).flatten().astype('float32' if task_type == 'regression' else 'int64')



X_all = data.iloc[:, 1:].astype('float32')
y_all = np.array(data.iloc[:, 0].astype('float32' if task_type == 'regression' else 'int64'))
if task_type != 'regression':
    # y_all = sklearn.preprocessing.LabelEncoder().fit_transform(y_all).astype('int64')
    y_train = sklearn.preprocessing.LabelEncoder().fit_transform(y_train).astype('int64')
    y_val = sklearn.preprocessing.LabelEncoder().fit_transform(y_val).astype('int64')
    y_test = sklearn.preprocessing.LabelEncoder().fit_transform(y_test).astype('int64')
n_classes = int(max(y_all)) + 1 if task_type == 'multiclass' else None


X = {}
y = {}
# X['train'], X['test'], y['train'], y['test'] = sklearn.model_selection.train_test_split(
#     X_all, y_all, train_size=0.8
# )
# X['train'], X['val'], y['train'], y['val'] = sklearn.model_selection.train_test_split(
#     X['train'], y['train'], train_size=0.8
# )
X['train'], X['val'], X['test']  = x_train, x_val, x_test
y['train'], y['val'], y['test']  = y_train, y_val, y_test

# lds = True # True False
# reweight = 'sqrt_inv'
# lds_kernel = 'gaussian'
# lds_ks = 5
# lds_sigma = 2

lds = args.lds
reweight = args.reweight
lds_kernel = args.lds_kernel
lds_ks = args.lds_ks
lds_sigma = args.lds_sigma

ldsModule = LDS(y['train'])
if reweight == "none":
    weights = None
else:
    weights = ldsModule._prepare_weights(
        reweight=reweight, 
        max_target = 5, 
        lds=lds, 
        lds_kernel=lds_kernel, 
        lds_ks=lds_ks, 
        lds_sigma=lds_sigma)
# weights = ldsModule._prepare_weights(
#     reweight=reweight, 
#     max_target = 5, 
#     lds=lds, 
#     lds_kernel=lds_kernel, 
#     lds_ks=lds_ks, 
#     lds_sigma=lds_sigma)
# weights = None
# not the best way to preprocess features, but enough for the demonstration
# preprocess = sklearn.preprocessing.StandardScaler().fit(X['train'])
# X = {
#     k: torch.tensor(preprocess.transform(v), device=device)
#     for k, v in X.items()
# }
X = {k: torch.tensor(np.array(v), device=device) for k, v in X.items()}
y = {k: torch.tensor(v, device=device) for k, v in y.items()}

# !!! CRUCIAL for neural networks when solving regression problems !!!
# if task_type == 'regression':
#     y_mean = y['train'].mean().item()
#     y_std = y['train'].std().item()
#     y = {k: (v - y_mean) / y_std for k, v in y.items()}
# else:
#     y_std = y_mean = None

if task_type != 'multiclass':
    y = {k: v.float() for k, v in y.items()}
#%%
d_out = n_classes or 1
# model = FTTransformer.make_default(
#     n_num_features=X_all.shape[1],
#     cat_cardinalities=None,
#     last_layer_query_idx=[-1],  # it makes the model faster and does NOT affect its output
#     d_out=d_out
# )
model = FTTransformer.make_default(
    n_num_features=X_all.shape[1],
    cat_cardinalities=args.cat_cardinalities,
    last_layer_query_idx=[-1],  # it makes the model faster and does NOT affect its output
    d_out=d_out,
    fds=args.fds,
    bucket_num=args.bucket_num,
    bucket_start=args.bucket_start,
    start_update=args.start_update,
    start_smooth=args.start_smooth,
    kernel=args.fds_kernel,
    ks=args.fds_ks,
    sigma=args.fds_sigma,
    momentum=args.fds_mmt
)
#%%
model.to(device)
# lr = 0.001
# weight_decay = 0.0
lr = args.lr
weight_decay = args.weight_decay
optimizer = (
    model.make_default_optimizer()
    if isinstance(model, FTTransformer)
    else torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
)
loss_fn = (
    F.binary_cross_entropy_with_logits
    if task_type == 'binclass'
    else F.cross_entropy
    if task_type == 'multiclass'
    else F.mse_loss
)
#%%
def apply_model(x_num, x_cat=None, targets=None, epoch=None):
    if isinstance(model, FTTransformer):
        return model(x_num, x_cat, targets, epoch)
    elif isinstance(model, (MLP, ResNet)):
        assert x_cat is None
        return model(x_num)
    else:
        raise NotImplementedError(
            f'Looks like you are using a custom model: {type(model)}.'
            ' Then you have to implement this branch first.'
        )


@torch.no_grad()
def evaluate(part):
    model.eval()
    prediction = []
    for batch in zero.iter_batches(X[part], 1024):
        res, _ = apply_model(batch)
        prediction.append(res)
    prediction = torch.cat(prediction).squeeze(1).cpu().numpy()
    target = y[part].cpu().numpy()

    if task_type == 'binclass':
        prediction = np.round(scipy.special.expit(prediction))
        score = sklearn.metrics.accuracy_score(target, prediction)
    elif task_type == 'multiclass':
        prediction = prediction.argmax(1)
        score = sklearn.metrics.accuracy_score(target, prediction)
    else:
        assert task_type == 'regression'
        score = sklearn.metrics.mean_squared_error(target, prediction)#** 0.5 * y_std
    return score, target, prediction
#%%
# batch_size = 256
batch_size = args.batch_size
train_loader = zero.data.IndexLoader(len(X['train']), batch_size, device=device)
progress = zero.ProgressTracker(patience=100)
_s, _, _ = evaluate("test")
print(f'Test score before training: {_s:.4f}')
#%%
# fds = True
# start_update = 0
# n_epochs = 2
fds = args.fds
start_update = args.start_update
n_epochs = args.epoch
report_frequency = len(X['train']) // batch_size // 2
for epoch in range(1, n_epochs + 1):
    for iteration, batch_idx in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()
        x_batch = X['train'][batch_idx]
        y_batch = y['train'][batch_idx]
        w_batch = torch.Tensor(weights)[batch_idx] if weights is not None else torch.Tensor([np.float32(1.)])
        # loss = loss_fn(apply_model(x_batch).squeeze(1), y_batch)
        res, _ = apply_model(x_batch, epoch=epoch, targets=y_batch)
        loss = ldsModule.weighted_mse_loss(res.squeeze(1), y_batch, w_batch)
        loss.backward()
        optimizer.step()
        if iteration % report_frequency == 0:
            print(f'(epoch) {epoch} (batch) {iteration} (loss) {loss.item():.4f}')

    if fds and epoch >= start_update:
        # print(f"Create Epoch [{epoch}] features of all training data...")
        encodings, labels = [], []
        with torch.no_grad():
            for batch_idx in train_loader:
                # inputs = inputs.cuda(non_blocking=True)
                # inputs = inputs.to(torch.float32)
                x_batch = X['train'][batch_idx]
                y_batch = y['train'][batch_idx]
                res, feature = apply_model(x_batch, epoch=epoch, targets=y_batch)
                
                encodings.extend(feature.data.squeeze().cpu().numpy())
                labels.extend(y_batch.data.squeeze().cpu().numpy())

        # encodings, labels = torch.from_numpy(np.vstack(encodings)).cuda(), torch.from_numpy(np.hstack(labels)).cuda()
        encodings, labels = torch.from_numpy(np.vstack(encodings)), torch.from_numpy(np.hstack(labels))
        model.transformer.head.FDS.update_last_epoch_stats(epoch)
        model.transformer.head.FDS.update_running_stats(encodings, labels, epoch)

    val_score, _, _ = evaluate('val')
    test_score, real, pred = evaluate('test')
    print(f'Epoch {epoch:03d} | Validation score: {val_score:.4f} | Test score: {test_score:.4f}', end='')
    progress.update((-1 if task_type == 'regression' else 1) * val_score)
    if progress.success:
        print(' <<< BEST VALIDATION EPOCH', end='')
    print()
    if progress.fail:
        break

show_fig = True
if show_fig:
    plt.plot(real, label='True values in {} set'.format("TEST"))
    plt.plot(pred, label='Pred. values in {} set'.format("TEST"))
    plt.legend()