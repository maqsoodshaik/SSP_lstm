#!/usr/bin/env python3
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import os
import csv
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
import fusion
#set random seeds
seed=42
torch.manual_seed(seed)
writer = SummaryWriter()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
# input_size = 784 # 28x28
num_classes = 1
num_epochs = 2
batch_size = 2
learning_rate = 0.0001
loss_hyp = 0.3
input_size = 256
sequence_length = 300
hidden_size = 2000
num_layers = 3


# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                           batch_size=batch_size,
#                                           shuffle=False)
org_path = os.getcwd()
#read csv
rows = []
rows_test = []
with open("bc_detection_train.csv", 'r') as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    for row in csvreader:
        rows.append(row)
with open("bc_detection_val.csv", 'r') as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    for row in csvreader:
        rows_test.append(row)
print(header)
print(rows_test[:][0])
#reading input data
path = os.getcwd() + '/' + 'resnet50'
dir_list = os.listdir(path)
os.chdir(path)



print(os.getcwd())
# combined_data = np.array([np.load(fname) for fname in dir_list])
combined_data_video = np.array([np.load(fname[0]+'_video.npy') for fname in rows])
print(os.getcwd())
combined_data_audio = np.array([np.load(fname[0]+'_audio.npy') for fname in rows])
labels = np.array([np.array(fname[1], dtype=np.float16) for fname in rows])



combined_data_test_video = np.array([np.load(fname[0]+'_video.npy') for fname in rows_test])
combined_data_test_audio = np.array([np.load(fname[0]+'_audio.npy') for fname in rows_test])
labels_test = np.array([np.array(fname[1], dtype=np.float16) for fname in rows_test])
os.chdir(org_path)
tensor_x_video = torch.Tensor(combined_data_video) # transform to torch tensor
tensor_x_audio = torch.Tensor(combined_data_audio) # transform to torch tensor





tensor_x_test_video = torch.Tensor(combined_data_test_video) # transform to torch tensor
tensor_x_test_audio = torch.Tensor(combined_data_test_audio) # transform to torch tensor
tensor_y_test = torch.Tensor(labels_test.astype(np.float64))



my_dataset_video_test = TensorDataset(tensor_x_test_video,tensor_y_test)
my_dataset_audio_test = TensorDataset(tensor_x_test_audio,tensor_y_test)

tensor_y = torch.Tensor(labels.astype(np.float64))
my_dataset_video = TensorDataset(tensor_x_video,tensor_y)
my_dataset_audio = TensorDataset(tensor_x_audio,tensor_y)
dataset_train_video, dataset_validate_video = train_test_split(
        my_dataset_video, test_size=0.5, random_state=84 #0.02
    )
dataset_train_audio, dataset_validate_audio = train_test_split(
        my_dataset_audio, test_size=0.5, random_state=84 #0.02
    )

#---------------------
class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

my_dataloader = torch.utils.data.DataLoader(
             ConcatDataset(
                 dataset_train_video,
                 dataset_train_audio
             ),
             batch_size=batch_size, shuffle=True)
my_dataloader_val = torch.utils.data.DataLoader(
             ConcatDataset(
                 dataset_validate_video,
                 dataset_validate_audio
             ),
             batch_size=batch_size)
my_dataloader_test = torch.utils.data.DataLoader(
             ConcatDataset(
                 my_dataset_video_test,
                 my_dataset_audio_test
             ),
             batch_size=batch_size)
#---------------------




class SUBNET(nn.Module):
    def __init__(self,num_classes):
      super(SUBNET, self).__init__()
      self.dropout1 = nn.Dropout2d(0.25)
      self.dropout2 = nn.Dropout2d(0.5)
      self.fc1 = nn.Linear(2048, 512)
      self.fc2 = nn.Linear(512, 128)
      self.fc = nn.Linear(128, num_classes)
      self.relu = nn.ReLU()
    # x represents our data
    def forward(self, x):
      # Pass data through conv1
      x = self.dropout1(x)
      # Flatten x with start_dim=1
      x = torch.flatten(x, 1)
      # Pass data through fc1
      x = self.fc1(x)
      x = self.relu(x)
      x = self.dropout2(x)
      x = self.fc2(x)
      x_s = self.fc(x)
      # Apply softmax to x
    #   output = x.reshape(300,128)

      return x, torch.sigmoid(x_s)




#Fully connected neural network with one hidden layer
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        #self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # -> x needs to be: (batch_size, seq, input_size)

        # or:
        #self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # self.fc_enc = SUBNET()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # x: (n, 28, 28), h0: (2, n, 128)

        # Forward propagate RNN
        #out, _ = self.rnn(x, h0)
        # or:
        # x = self.fc_enc(x)
        out, _ = self.lstm(x, (h0,c0))

        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out: (n, 28, 128)

        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        # out: (n, 128)

        out = self.fc(out)
        # out: (n, 10)
        return torch.sigmoid(out)

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
model_trans = fusion.TransformerEncoder(num_layers = 4,input_dim =128,num_heads =4, dim_feedforward = 256)
model_fc = SUBNET(num_classes).to(device)
# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion_fc = nn.BCELoss()
criterion_trans = nn.BCELoss()
# optimizer_fc = torch.optim.Adam(model_fc.parameters(), lr=learning_rate)
# Train the model
best_val_acc = 0
for epoch in range(num_epochs):
    correct = 0
    num_samples = 0
    model.train()
    model_fc.train()
    model_trans.train()
    for i, (dataset1, dataset2) in enumerate(my_dataloader):
        # origin shape: [N, 1, 28, 28]
        # resized: [N, 300, 2048][N,300,128]
        data1 = dataset1[0].to(device)
        labels = dataset1[1].to(device)
        data2 = dataset2[0].to(device)
        label2 = dataset2[1].to(device)
         # Forward pass
        images_i = torch.as_tensor([])
        images = torch.as_tensor([])
        loss_fc = 0
        loss_trans = 0
        for k in range(data1.size(0)):
            images_o,outputs_fc_s = model_fc(data1[k])
            loss_fc += criterion_fc(outputs_fc_s, labels[k].reshape(1,1).expand(sequence_length, 1))
            audio_i=data2[k]
            audio_i = audio_i.unsqueeze(1)
            images_i=images_o 
            images_i = images_i.unsqueeze(1)
            aggreg = torch.cat((audio_i,images_i),1)
            out,out_s = model_trans(aggreg)
            out = out.reshape(sequence_length,input_size)
            out_s = out_s.reshape(sequence_length,2)
            loss_trans += criterion_fc(out_s, labels[k].reshape(1,1).expand(sequence_length, 2))
            images = torch.cat((images,out))

        
        images = images.reshape(-1, sequence_length, input_size).to(device)
        # print(images.shape)
        labels = labels.to(device)
        num_samples+=labels.size(0)
        # Forward pass
        outputs = model(images)
        # print(f'outputs.shape:{outputs.shape}')
        # print(labels.shape)
        # outputs = outputs.squeeze()
        loss = criterion(outputs, labels.unsqueeze(1))
        loss +=loss_hyp*(loss_fc+loss_trans)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        predicted = (outputs > 0.5).long()
        # print(f'predicted.shape:{predicted.shape}')
        # print(f'labels.shape:{labels.shape}')
        # print(f'correct_pre:{correct}')
        correct += (predicted.squeeze()== labels).sum().item()
       
    print('[%d/%d] loss: %.3f, accuracy: %.3f' %
          (i , epoch, loss.item(), 100 * correct /num_samples))
    writer.add_scalars('Loss',{'train':loss.item()},epoch)
    writer.add_scalars('Accuracy', {'train': 100 * correct /num_samples},epoch)
        
    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    num_samples_val = 0
    model.eval()
    model_fc.eval()
    model_trans.eval()
    with torch.no_grad():
        correct_val = 0
        for i, (dataset1, dataset2) in enumerate(my_dataloader_val):
            #####

            data1 = dataset1[0].to(device)
            labels = dataset1[1].to(device)
            data2 = dataset2[0].to(device)
            label2 = dataset2[1].to(device)
            # Forward pass
            images = torch.as_tensor([])
            loss_fc = 0
            for k in range(data1.size(0)):
                images_o,outputs_fc_s = model_fc(data1[k])
            
                # loss_fc += criterion_fc(outputs_fc_s, labels.unsqueeze(1).expand(sequence_length, 1))
                audio_i=data2[k]
                audio_i = audio_i.unsqueeze(1)
                images_i=images_o 
                images_i = images_i.unsqueeze(1)
                aggreg = torch.cat((audio_i,images_i),1)
                out,out_s = model_trans(aggreg)
                out = out.reshape(sequence_length,input_size)
                out_s = out_s.reshape(sequence_length,2)
                # loss_trans += criterion_fc(out_s, labels[k].reshape(1,1).expand(sequence_length, 2))
                images = torch.cat((images,out))

            images = images.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)
            num_samples_val+=labels.size(0)
            outputs = model(images)
            predicted = (outputs > 0.5).long()
            correct_val += (predicted.squeeze()== labels).sum().item()

        val_acc = 100 * correct_val / num_samples_val
        print(f'Accuracy of the network on the validation: {val_acc} %')
        writer.add_scalars('Accuracy', {'val': val_acc},epoch)
    if(val_acc> best_val_acc):
        best_val_acc = val_acc
        torch.save(model.state_dict(),'./best_model'+'.ckpt')                         
        print("best model with val acc "+ str(best_val_acc)+ "is saved")
model.eval()
model_fc.eval()
model_trans.eval()
model.load_state_dict(torch.load('./best_model.ckpt'))   
with torch.no_grad():
        correct_val = 0
        num_samples_val = 0
        for i, (dataset1, dataset2) in enumerate(my_dataloader_val):
            #####

            data1 = dataset1[0].to(device)
            labels = dataset1[1].to(device)
            data2 = dataset2[0].to(device)
            label2 = dataset2[1].to(device)
            # Forward pass
            images = torch.as_tensor([])
            loss_fc = 0
            for k in range(data1.size(0)):
                images_o,outputs_fc_s = model_fc(data1[k])
            
                audio_i=data2[k]
                audio_i = audio_i.unsqueeze(1)
                images_i=images_o 
                images_i = images_i.unsqueeze(1)
                aggreg = torch.cat((audio_i,images_i),1)
                out,out_s = model_trans(aggreg)
                out = out.reshape(sequence_length,input_size)
                out_s = out_s.reshape(sequence_length,2)
                # loss_trans += criterion_fc(out_s, labels[k].reshape(1,1).expand(sequence_length, 2))
                images = torch.cat((images,out))

            images = images.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)
            num_samples_val+=labels.size(0)
            outputs = model(images)
            predicted = (outputs > 0.5).long()
            correct_val += (predicted.squeeze()== labels).sum().item()

        val_acc = 100 * correct_val / num_samples_val
        print(f'Accuracy of the network on the test: {val_acc} %')
