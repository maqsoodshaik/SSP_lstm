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
#set random seeds
seed=42
torch.manual_seed(seed)
writer = SummaryWriter()

# Device configuration
device = torch.device('cuda')

# Hyper-parameters 
# input_size = 784 # 28x28
num_classes = 1
num_epochs = 30
batch_size = 64
learning_rate = 0.0001

input_size = 2048
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
print(header)
print(rows[:][0])

with open("bc_detection_val.csv", 'r') as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    for row in csvreader:
        rows_test.append(row)
print(header)
print(rows_test[:][0])


#reading input data
#path = os.getcwd() + '/' + 'resnet50'
path = '/content/drive/MyDrive/video_features'
dir_list = os.listdir(path)
os.chdir(path)




# combined_data = np.array([np.load(fname) for fname in dir_list])
combined_data = np.array([np.load(fname[0]+'_video.npy') for fname in rows])
labels = np.array([np.array(fname[1], dtype=np.float16) for fname in rows])

path = '/content/drive/MyDrive/video_feat_test'
dir_list = os.listdir(path)
os.chdir(path)

combined_data_test = np.array([np.load(fname[0]+'_video.npy') for fname in rows_test])
labels_test = np.array([np.array(fname[1], dtype=np.float16) for fname in rows_test])

os.chdir(org_path)

tensor_x = torch.Tensor(combined_data) # transform to torch tensor
tensor_y = torch.Tensor(labels.astype(np.float))


tensor_x_test = torch.Tensor(combined_data_test) # transform to torch tensor
tensor_y_test = torch.Tensor(labels_test.astype(np.float))

my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
dataset_train, dataset_validate = train_test_split(
        my_dataset, test_size=0.02, random_state=84
    )

my_dataset_test = TensorDataset(tensor_x_test,tensor_y_test)
print(tensor_x.shape)
print(tensor_y.shape)


my_dataloader = DataLoader(dataset_train,batch_size=batch_size) # create your dataloader
my_dataloader_val = DataLoader(dataset_validate,batch_size=batch_size) 
my_dataloader_test = DataLoader(my_dataset_test,batch_size=batch_size) 


# for i, (images, labels) in enumerate(my_dataloader): 
#   print(images.shape)
#   print(labels)

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

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Train the model
n_total_steps = len(my_dataloader)
best_val_acc = 0
for epoch in range(num_epochs):
    correct = 0
    num_samples = 0
    model.train()
    for i, (images, labels) in enumerate(my_dataloader):
        # origin shape: [N, 1, 28, 28]
        # resized: [N, 300, 2048][N,300,128]
        
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

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        predicted = (outputs > 0.5).long()
        # print(f'predicted.shape:{predicted.shape}')
        # print(f'labels.shape:{labels.shape}')
        # print(f'correct_pre:{correct}')
        correct += (predicted.squeeze()== labels).sum().item()
        # print(f'predicted:{predicted}')
        # print(f'correct:{correct}')
        # print(f'labels.size:{labels.size(0)}')
        # if (i+1) % 1 == 0:
        #     print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
        # print(f'correct:{correct}')
    # print(f'tensor_y.size(0):{tensor_y.size(0)}')
    print('[%d/%d] loss: %.3f, accuracy: %.3f' %
          (i , epoch, loss.item(), 100 * correct /num_samples))
    writer.add_scalars('Loss',{'train':loss.item()},epoch)
    writer.add_scalars('Accuracy', {'train': 100 * correct /num_samples},epoch)
        
    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    num_samples_val = 0
    model.eval()
    with torch.no_grad():
        correct_val = 0
        for i, (images, labels) in enumerate(my_dataloader_val):
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
model.load_state_dict(torch.load('/content/best_model.ckpt'))   
with torch.no_grad():
        correct_val = 0
        num_samples_val = 0
        for i, (images, labels) in enumerate(my_dataloader_test):
            images = images.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)
            num_samples_val+=labels.size(0)
            outputs = model(images)
            predicted = (outputs > 0.5).long()
            correct_val += (predicted.squeeze()== labels).sum().item()

        val_acc = 100 * correct_val / num_samples_val
        print(f'Accuracy of the network on the test: {val_acc} %')