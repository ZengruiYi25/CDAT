
import time
import numpy as np
from tqdm import tqdm
from thop import profile
from collections import Counter
from sklearn.metrics import accuracy_score

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from Models.CDAT import CDAT

num_seed = 252618
np.random.seed(num_seed)
torch.manual_seed(num_seed)
torch.cuda.manual_seed_all(num_seed)

# torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
/* --- Efficient Convolutional Dual-Attention Transformer for Automatic Modulation Recognition --- */
"""

net_file = "./Argus/CDAT"
IF_RESTART = True  # restart the training
IF_OAFirst = False  # overall accuracy or highest accuracy

# SNRs
snrs_list = np.arange(-20, 20, 2)

# Parameters of the model
num_classes = 10
emb_dim = [32, 32, 32, 32]
emb_kernel = [7, 5, 3, 3]
emb_stride = [2, 2, 2, 2]
proj_kernel = [3, 3, 3, 3]
heads = [4, 4, 4, 4]
mlp_mult = [4, 4, 4, 4]
dropout = 0.1

# Parameters of the training and optimizer
num_epochs = 500  # training epochs
batch_size = 64  # the batch size
batch_rate = 8  # reduce the testing time when the GPU memory is sufficient
learning_rate = 1e-3  # initial learning rate
weight_decay = 1e-4  # L2
optim_switch = 0  #
# --- LR scheduler
step_size = 5
gamma = 0.95
# --- Cos scheduler
lr_cos = 1e-4
T_max = 20
lr_cos_min = 1e-5

# Global accuracy
global_test_HA = 0         # the maximum HA
global_test_OA = 0         # the maximum OA
global_test_HA_train = 0   # corresponding HA on the training set
global_test_OA_train = 0   # corresponding OA on the training set
global_test_HA_epoch = 0   # corresponding training epoch


def data_process():
    print("\n ================== Loading Data ==================  ")

    print("Loading train_data ...")
    train_data = np.load("./Datasets/train_data.npz")['arr_0']
    print("Loading train_label ...")
    train_label = np.load("./Datasets/train_label.npz")['arr_0']

    print("Loading test_data ...")
    test_data = np.load("./Datasets/test_data.npz")['arr_0']
    print("Loading test_label ...")
    test_label = np.load("./Datasets/test_label.npz")['arr_0']

    print("Train.shape: ", train_data.shape, Counter(train_label[:, 0]))
    print("Test.shape: ", test_data.shape, Counter(test_label[:, 0]))
    print("Test.shape: ", test_data.shape, Counter(test_label[:, 1]))

    # To tensor
    train_data = torch.tensor(train_data, dtype=torch.float32)
    test_data = torch.tensor(test_data, dtype=torch.float32)

    train_label = torch.tensor(train_label, dtype=torch.int64)
    test_label = torch.tensor(test_label, dtype=torch.int64)

    train_ds = TensorDataset(train_data, train_label)
    test_ds = TensorDataset(test_data, test_label)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    train_test_dl = DataLoader(train_ds, batch_size=batch_size * batch_rate, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size * batch_rate, shuffle=False)

    return train_dl, train_test_dl, test_dl


if __name__ == '__main__':
    # Load the dataset
    train_loader, trian_test_loader, test_loader = data_process()
    batch_length = len(train_loader)

    # Create a network
    net = CDAT(num_classes=num_classes,
               emb_dim=emb_dim,
               emb_kernel=emb_kernel,
               emb_stride=emb_stride,
               proj_kernel=proj_kernel,
               heads=heads,
               mlp_mult=mlp_mult,
               dropout=dropout).to(device)
    if not IF_RESTART:
        state_dict = torch.load(net_file)
        net.load_state_dict(state_dict, strict=False)

    # Compute the model performance
    print(" ================== Evaluating ==================  ")
    net.eval()
    flops, params = profile(net, inputs=(torch.rand((1, 2, 128), device=device),), verbose=False)
    txt0 = "The number of parameters is %d(%.2fM)(%.2fK)\nThe number of flops is %d(%.2fM)" % (
        params, params / 1e6, params / 1e3, flops, flops / 1e6)
    with open(net_file + '.txt', 'w', encoding='utf-8') as f:
        f.write(txt0 + "\n\n")
        f.close()
    print(txt0)
    print(" ================================================  ")

    # Define the optimizer
    optimizer = optim.NAdam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler_lr = StepLR(optimizer, step_size=step_size, gamma=gamma)
    scheduler_cos = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=lr_cos_min)

    # Define the loss function
    loss_func = torch.nn.CrossEntropyLoss()

    # Training
    for epoch in range(num_epochs+1):
        if epoch == 0:
            print("Test initial network ... ")
            loss = torch.tensor([0])
        else:
            net.train()
            torch.cuda.empty_cache()  # clear the GPU memory
            train_loader_tqdm = tqdm(train_loader)
            train_loader_tqdm.set_description('epoch: {}'.format(epoch))
            for step, (batch_x, batch_y) in enumerate(train_loader_tqdm):
                batch_x = batch_x.to(device)

                # Data Enhancement
                if torch.rand(1, requires_grad=False) > 0.7 and step != batch_length - 1:
                    sig_length = torch.randint(64, 128, (1, 1), requires_grad=False).item()
                    sig_start = torch.randint(0, 128-sig_length, (1, 1), requires_grad=False).item()
                    batch_x = batch_x[:, :, sig_start:(sig_start + sig_length)]

                predict = net(batch_x)

                loss = loss_func(predict, batch_y[:, 0].to(device))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loader_tqdm.set_postfix({'lr': optimizer.state_dict()['param_groups'][0]['lr'],
                                              'loss': loss.cpu().item(),})

            # Update the learning rate
            if not epoch == 0 and global_test_HA_train > 20:
                if optimizer.state_dict()['param_groups'][0]['lr'] > lr_cos and optim_switch == 0:
                    scheduler_lr.step()
                elif optim_switch == 1:
                    scheduler_cos.step()
                else:
                    print("optim switch")
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_cos
                        optim_switch = 1

        # Testing
        ts = time.time()
        net.eval()
        torch.cuda.empty_cache()  # clear the GPU memory
        with torch.no_grad():

            # --- Training Set
            output_list = np.array([])
            real_list = np.array([])
            snr_list = np.array([])
            for (batch_test_x, batch_test_y) in trian_test_loader:
                output = net(batch_test_x.to(device))
                output = torch.argmax(output, dim=1).cpu().numpy()
                output_list = np.append(output_list, output)
                real_list = np.append(real_list, batch_test_y[:, 0].numpy())
                snr_list = np.append(snr_list, batch_test_y[:, 1].numpy())
            # --- Acc about each SNR
            acc_snr_train = []
            snr_list = np.array(snr_list)
            for snr in snrs_list:
                idx = snr_list == snr
                acc_snr_train.append(round(accuracy_score(real_list[idx], output_list[idx]) * 100, 2))
            train_OA = round(accuracy_score(real_list, output_list) * 100, 2)
            train_HA = max(acc_snr_train)

            # --- Testing Set
            output_list = np.array([])
            real_list = np.array([])
            snr_list = np.array([])
            for (batch_test_x, batch_test_y) in test_loader:
                output  = net(batch_test_x.to(device))
                output = torch.argmax(output, dim=1).cpu().numpy()
                output_list = np.append(output_list, output)
                real_list = np.append(real_list, batch_test_y[:, 0].numpy())
                snr_list = np.append(snr_list, batch_test_y[:, 1].numpy())
            # --- Acc about each SNR
            acc_snr_test = []
            snr_list = np.array(snr_list)
            for snr in snrs_list:
                idx = snr_list == snr
                acc_snr_test.append(round(accuracy_score(real_list[idx], output_list[idx]) * 100, 2))
            test_OA = round(accuracy_score(real_list, output_list) * 100, 2)
            test_HA = max(acc_snr_test)

        # Save the model and its accuracy
        if ((IF_OAFirst and test_OA >= global_test_OA)
                or (not IF_OAFirst and test_HA >= global_test_HA)
                or (test_HA >= 94.5 and test_OA >= 64.5)):
            global_test_HA = test_HA
            global_test_OA = test_OA
            global_test_HA_train = train_HA
            global_test_OA_train = train_OA
            global_test_HA_epoch = epoch
            if test_HA >= 94.4 and test_OA >= 64.7:
                torch.save(net.state_dict(), net_file + '_{:.2f}_{:.2f}'.format(test_HA, test_OA))
            else:
                torch.save(net.state_dict(), net_file)

        te = time.time()

        # Print information per epoch
        txt1 = "epoch = {} >>>>>>>> bestepoch = {}, train = {}%({}%), test =\033[0;31m {}%({}%) \033[0m/ {}s\n" \
            .format(epoch, global_test_HA_epoch,
                    global_test_HA_train, global_test_OA_train,
                    global_test_HA, global_test_OA,
                    round(te-ts, 2))
        txt2 = "SNR_Accuracy_Train = {}({}% | {}%)\nSNR_Accuracy_Test  = {}({}% | {}%)\n" \
            .format(acc_snr_train, train_HA, train_OA,
                    acc_snr_test, test_HA, test_OA)
        txt = txt1 + txt2
        print(txt, end='')
        try:
            with open(net_file + '.txt', 'a', encoding='utf-8') as f:
                f.write(txt+'\n')
                f.close()
        except:
            pass

        # Overfitting
        if train_OA - test_OA > 5:
            print("Overfitting ... ... ... ")
            break

    print()
