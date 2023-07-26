import numpy as np
import torch

# import functional torch as f
import torch.nn as nn
from torch.nn import functional as F
from data.synthetic_dataset import create_synthetic_dataset, SyntheticDataset, CustomDataset, Data_utility, TrafficDataset
from models.seq2seq import EncoderRNN, DecoderRNN, Net_GRU
from loss.dilate_loss import dilate_loss
from torch.utils.data import DataLoader
import random
from tslearn.metrics import dtw, dtw_path
import matplotlib.pyplot as plt
import warnings
import warnings; warnings.simplefilter('ignore')
import pdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random.seed(0)

# parameters
batch_size = 256
N = 500
N_input = 168#12#20
N_output = 24#6#20  
sigma = 0.01
gamma = 0.01

Data = Data_utility("data/traffic.txt", 0.6, 0.2, False, 24, 168, True);
Data.train[0] = Data.train[0][:,:,0]
Data.train[1] = Data.train[1][:,:,0]
Data.valid[0] = Data.valid[0][:,:,0]
Data.valid[1] = Data.valid[1][:,:,0]
Data.test[0] = Data.test[0][:,:,0]
Data.test[1] = Data.test[1][:,:,0]
dataset_train = TrafficDataset(Data.train[0], Data.train[1])
dataset_valid = TrafficDataset(Data.valid[0], Data.valid[1])
dataset_test = TrafficDataset(Data.test[0], Data.test[1])


# Load synthetic dataset
# X_train_input,X_train_target,X_test_input,X_test_target,train_bkp,test_bkp = create_synthetic_dataset(N,N_input,N_output,sigma)
# dataset_train = SyntheticDataset(X_train_input,X_train_target, train_bkp)
# dataset_test  = SyntheticDataset(X_test_input,X_test_target, test_bkp)
#dataset_train = CustomDataset('data/train.npy')
#dataset_test = CustomDataset('data/test.npy')
trainloader = DataLoader(dataset_train, batch_size=batch_size,shuffle=True, num_workers=1, drop_last=True)
validloader = DataLoader(dataset_valid, batch_size=batch_size,shuffle=False, num_workers=1, drop_last=True)
testloader  = DataLoader(dataset_test, batch_size=batch_size,shuffle=True, num_workers=1, drop_last=True)

class DifferentialDivergenceLoss(nn.Module):
    def __init__(self, tau=0.1, epsilon=1e-8, lambda_reg=1, wght = 10):
        super(DifferentialDivergenceLoss, self).__init__()
        self.tau = tau
        self.epsilon = epsilon
        self.lambda_reg = lambda_reg
        self.wght = wght

    def forward(self, pred, true):
        mse_loss = F.mse_loss(pred, true)
        std_loss = F.mse_loss(torch.std(pred, dim=1), torch.std(true, dim=1))

        pred_diff = pred[:, 1:] - pred[:, :-1]
        true_diff = true[:, 1:] - true[:, :-1]
        #pred_diff = pred_diff.view(pred_diff.shape[0], pred_diff.shape[1], -1)
        #true_diff = true_diff.reshape(true_diff.shape[0], true_diff.shape[1], -1)

        #pred_prob = F.softmax(pred_diff / self.tau, dim=2)
        #true_prob = F.softmax(true_diff / self.tau, dim=2)
        reg_mse = F.mse_loss(pred_diff, true_diff)
        reg_std = F.mse_loss(torch.std(pred_diff, dim=1), torch.std(true_diff,dim=1))
        #reg_loss = torch.sum(pred_prob * torch.log((pred_prob + self.epsilon) / (true_prob + self.epsilon)))

        total_loss = mse_loss + 10*reg_mse + 10*reg_std + self.wght * std_loss

        return total_loss, mse_loss, reg_mse, self.lambda_reg * reg_std, self.wght * std_loss

def train_model(net,loss_type, learning_rate, epochs=1000, gamma = 0.001,
                print_every=50,eval_every=50, verbose=1, Lambda=1, alpha=0.8):
    
    optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)
    #criterion = torch.nn.MSELoss()

    criterion = DifferentialDivergenceLoss()#torch.nn.MSELoss()
    
    best_loss = 100000
    for epoch in range(epochs): 
        for i, data in enumerate(trainloader, 0):
            inputs, target, _ = data
            inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
            target = torch.tensor(target, dtype=torch.float32).to(device)
            batch_size, N_output = target.shape[0:2]                     

            # forward + backward + optimize
            outputs = net(inputs)
            loss_mse,loss_shape,loss_temporal = torch.tensor(0),torch.tensor(0),torch.tensor(0)
            
            if (loss_type=='mse'):
                loss_mse, l1,l2,l3,l4 = criterion(target,outputs)
                loss = loss_mse                   
 
            if (loss_type=='dilate'):    
                loss, loss_shape, loss_temporal = dilate_loss(target,outputs,alpha, gamma, device)             
                  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()          
        
        if(verbose):
            if (epoch % print_every == 0):
                print('epoch ', epoch, ' loss ',loss.item(),' loss shape ',loss_shape.item(),' loss temporal ',loss_temporal.item())
                eval_loss = eval_model(net,validloader,loss_type, gamma,verbose=1)
                # use eval_loss to save the best model
                if (eval_loss < best_loss):
                    print ("New best")
                    # save model using torch
                    torch.save(net.state_dict(), 'best_model.pt')
                    best_loss = eval_loss
    
    # load best model 'best_model.pt' using torch
    print ("Loading best model")
    net.load_state_dict(torch.load('best_model.pt'))
                                                 


def eval_model(net,loader, loss_type, gamma,verbose=1):   
    criterion = torch.nn.MSELoss()
    losses = []
    losses_mse = []
    losses_dtw = []
    losses_tdi = []   
    diff_loss = DifferentialDivergenceLoss()#torch.nn.MSELoss()
    for i, data in enumerate(loader, 0):
        loss_mse, loss_dtw, loss_tdi = torch.tensor(0),torch.tensor(0),torch.tensor(0)
        # get the inputs
        inputs, target, breakpoints = data
        inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
        target = torch.tensor(target, dtype=torch.float32).to(device)
        batch_size, N_output = target.shape[0:2]
        outputs = net(inputs)
        
        if (loss_type=='mse'):
                loss_mse, l1,l2,l3,l4 = diff_loss(target,outputs)
                loss = loss_mse                   
 
        if (loss_type=='dilate'):    
                loss, loss_shape, loss_temporal = dilate_loss(target,outputs,alpha=0.8, gamma=0.001, device=device)    
        # MSE    
        loss_mse = criterion(target,outputs)    
        loss_dtw, loss_tdi = 0,0
        # DTW and TDI
        for k in range(batch_size):         
            target_k_cpu = target[k,:,0:1].view(-1).detach().cpu().numpy()
            output_k_cpu = outputs[k,:,0:1].view(-1).detach().cpu().numpy()

            path, sim = dtw_path(target_k_cpu, output_k_cpu)   
            loss_dtw += sim
                       
            Dist = 0
            for i,j in path:
                    Dist += (i-j)*(i-j)
            loss_tdi += Dist / (N_output*N_output)            
                        
        loss_dtw = loss_dtw /batch_size
        loss_tdi = loss_tdi / batch_size

        # print statistics
        losses_mse.append( loss_mse.item() )
        losses_dtw.append( loss_dtw )
        losses_tdi.append( loss_tdi )
        losses.append( loss.item())

    print( ' Eval mse= ', np.array(losses_mse).mean() ,' dtw= ',np.array(losses_dtw).mean() ,' tdi= ', np.array(losses_tdi).mean(),
           ' loss= ', np.array(losses).mean()) 
    return np.array(losses).mean()

encoder = EncoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1, batch_size=batch_size).to(device)
decoder = DecoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1,fc_units=16, output_size=1).to(device)
net_gru_mse = Net_GRU(encoder,decoder, N_output, device).to(device)
train_model(net_gru_mse,loss_type='mse',learning_rate=0.001, epochs=500, gamma=gamma, print_every=50, eval_every=50,verbose=1)
print ("Testing")
_ = eval_model(net_gru_mse,testloader, loss_type='mse', gamma=gamma)

encoder = EncoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1, batch_size=batch_size).to(device)
decoder = DecoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1,fc_units=16, output_size=1).to(device)
net_gru_dilate = Net_GRU(encoder,decoder, N_output, device).to(device)
train_model(net_gru_dilate,loss_type='dilate',learning_rate=0.001, epochs=500, gamma=gamma, print_every=50, eval_every=50,verbose=1)
print ("Testing")
_ = eval_model(net_gru_dilate,testloader, loss_type='dilate', gamma=gamma)

# Visualize results
gen_test = iter(testloader)
test_inputs, test_targets, _ = next(gen_test)

test_inputs  = torch.tensor(test_inputs, dtype=torch.float32).to(device)
test_targets = torch.tensor(test_targets, dtype=torch.float32).to(device)
criterion = torch.nn.MSELoss()

nets = [net_gru_mse,net_gru_dilate]



for ind in range(1,51):
    plt.figure()
    plt.rcParams['figure.figsize'] = (17.0,5.0)  
    k = 1
    for net in nets:
        pred = net(test_inputs).to(device)

        input = test_inputs.detach().cpu().numpy()[ind,:,:]
        target = test_targets.detach().cpu().numpy()[ind,:,:]
        preds = pred.detach().cpu().numpy()[ind,:,:]

        plt.subplot(1,3,k)
        plt.plot(range(0,N_input) ,input,label='input',linewidth=3)
        plt.plot(range(N_input-1,N_input+N_output), np.concatenate([ input[N_input-1:N_input], target ]) ,label='target',linewidth=3)   
        plt.plot(range(N_input-1,N_input+N_output),  np.concatenate([ input[N_input-1:N_input], preds ])  ,label='prediction',linewidth=3)       
        plt.xticks(range(0,40,2))
        plt.legend()
        k = k+1

    plt.show()
