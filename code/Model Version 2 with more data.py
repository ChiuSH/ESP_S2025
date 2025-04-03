#Data Loader
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class ProteinDataset(Dataset):
    def __init__(self, csv_file, sequence_length=250):
        """
        Args:
            csv_file (str): Path to the CSV file with protein sequences and modification labels.
            sequence_length (int): Maximum sequence length (padding shorter sequences).
        """
        # Load the CSV data
        self.data = pd.read_csv(csv_file)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the protein sequence and modification labels
        sequence = self.data.iloc[idx, 1]  # Protein sequence column
        labels = self.data.iloc[idx, 2]  # Modification labels column
        sequence = sequence.replace('\r','').replace('\n','').replace('"','').replace("'",'').replace(" ",'')
        labels = labels.replace('\r','').replace('\n','').replace('"','').replace("'",'').replace(" ",'')[:]
        print(f"label before processing {labels} and length: {len(labels)}")
        # Convert the sequence into a list of amino acid indices
        amino_acid_dict = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'E': 6, 'Q': 7, 'G': 8, 'H': 9, 'I': 10, 'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'S': 16, 'T': 17, 'W': 18, 'Y': 19, 'V': 20}
        
        # Convert sequence to indices
        sequence_indices = [amino_acid_dict[aa] for aa in sequence]
        
        # Padding if the sequence is shorter than the expected length
        #if len(sequence_indices) < self.sequence_length:
            #sequence_indices += [0] * (self.sequence_length - len(sequence_indices))  # Use 0 as padding
        print(f"sequence indices is: {sequence_indices} and length: {len(sequence_indices)}")
        # Convert modification labels to a list of integers
        labels = [int(label) for label in labels.split(',')]
        
        # Ensure the label length matches the sequence length
        #if len(labels) < self.sequence_length:
            #labels += [0] * (self.sequence_length - len(labels))  # Padding labels if necessary
        print(f"label indices is: {labels} and length: {len(labels)}")
        # Convert sequence and labels to tensors
        sequence_tensor = torch.tensor(sequence_indices).long()  # Long tensor for sequence indices
        labels_tensor = torch.tensor(labels).float()  # Float tensor for labels (1 or 0)

        # Reshape sequence to match the input shape for the CNN (channels, sequence length)
        sequence_tensor = sequence_tensor.unsqueeze(0)  # Add a channel dimension (1 channel for 21 amino acids)

        return sequence_tensor, labels_tensor

# Load the data into a DataLoader
dataset = ProteinDataset(csv_file="code_test_data.csv", sequence_length=33)  # Adjust sequence length based on your data
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
data_pred = ProteinDataset(csv_file="code_predict.csv", sequence_length=33)
test_loader = DataLoader(data_pred, batch_size=54, shuffle=True)
# Example: Inspecting the loaded data
for batch_idx, (sequences, labels) in enumerate(train_loader):
    print(f"Batch {batch_idx+1}:")
    print("Sequences:", sequences)
    print("Labels:", labels)














#CNN Model (original)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class PhosphoCNN(nn.Module):
    def __init__(self, sequence_length):
        super(PhosphoCNN, self).__init__()

        # First 1D convolutional layer: input channels = 21 (one-hot encoding size for each amino acid),
        # output channels = 64 (number of filters), kernel size = 3 (size of the convolutional window)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # Second 1D convolutional layer: output channels = 128
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        # Max pooling layer to downsample the sequence
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Fully connected layer
        self.fc1 = nn.Linear(1024 , 75)  # Adjust size depending on the sequence length
        self.fc2 = nn.Linear(75, 33)  # Output one prediction per amino acid position
        self.out = nn.Linear(33, 1)

    def forward(self, x):
        # Apply first convolution, ReLU activation, and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        
        # Apply second convolution, ReLU activation, and max pooling
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten the output from 2D to 1D (for fully connected layer)
        x = x.view(-1, 128 * (x.size(2)))  # Adjust the size based on sequence length
        
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Sigmoid activation for binary classification (phosphorylation or not)
        
        return x


# Example: Train the model

# Define the model
sequence_length = 250 # Example sequence length
model = PhosphoCNN(sequence_length)

# Loss function (Binary Cross-Entropy)
# Calculate class weights
class_weights = torch.tensor([20.0])  # Give 10 times more weight to '1' class
#criterion = nn.BCELoss(pos_weight=class_weights)
criterion = nn.BCEWithLogitsLoss(pos_weight = class_weights)

# Optimizer (Adam)
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50
# Example training loop
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    
    for inputs, labels in train_loader:  # Assume `train_loader` provides protein sequences and labels
        optimizer.zero_grad()
        inputs=inputs.float()
        # Forward pass
        outputs = model(inputs)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')



# Evaluate the model
model.eval()  # Set the model to evaluation mode

with torch.no_grad():
    for inputs, labels in test_loader:  # Assume `test_loader` provides test data

        inputs = inputs.float()
        outputs = model(inputs)
        predictions = (outputs > 0.5).float()  # Convert probabilities to binary predictions (0 or 1)
        print(f"predictions:\n{predictions}\ninputs:\n{labels}")
        # Calculate accuracy or other metrics
        correct = ((predictions==1)&(labels==1)).sum().item()
        print(f"number correct{correct}")
        total_ones_in_label = (labels==1).sum().item()
        total_ones_predicted = predictions.sum().item()
        false_positives = ((predictions == 1)&(labels==0)).sum().item()
        #the math below is wrong
        accuracy = correct/total_ones_in_label if total_ones_in_label > 0 else 0.0
        print(f'Accuracy: {correct/total_ones_predicted:.4f}\nfalse positives: {false_positives}\naccuracy/false_positive: {(correct/total_ones_predicted)/false_positives}')




#CNN Model(messed with parameters)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class PhosphoCNN(nn.Module):
    def __init__(self, sequence_length):
        super(PhosphoCNN, self).__init__()

        # First 1D convolutional layer: input channels = 21 (one-hot encoding size for each amino acid),
        # output channels = 64 (number of filters), kernel size = 3 (size of the convolutional window)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=4, stride=1, padding=1)
        
        # Second 1D convolutional layer: output channels = 128
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=12, stride=1, padding=1)
        
        # Max pooling layer to downsample the sequence
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Fully connected layer
        self.fc1 = nn.Linear(384 , 128)  # Adjust size depending on the sequence length
        self.fc2 = nn.Linear(128, 33)  # Output one prediction per amino acid position
        self.out = nn.Linear(33, 1)

    def forward(self, x):
        # Apply first convolution, ReLU activation, and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        
        # Apply second convolution, ReLU activation, and max pooling
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten the output from 2D to 1D (for fully connected layer)
        x = x.view(-1, 128 * (x.size(2)))  # Adjust the size based on sequence length
        
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Sigmoid activation for binary classification (phosphorylation or not)
        
        return x


# Example: Train the model

# Define the model
sequence_length = 33 # Example sequence length
model = PhosphoCNN(sequence_length)

# Loss function (Binary Cross-Entropy)
# Calculate class weights
class_weights = torch.tensor([50.0])  # Give 10 times more weight to '1' class
#criterion = nn.BCELoss(pos_weight=class_weights)
criterion = nn.BCEWithLogitsLoss(pos_weight = class_weights)

# Optimizer (Adam)
optimizer = optim.Adam(model.parameters(), lr=0.002)

epochs = 10
# Example training loop
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    
    for inputs, labels in train_loader:  # Assume `train_loader` provides protein sequences and labels
        optimizer.zero_grad()
        inputs=inputs.float()
        # Forward pass
        outputs = model(inputs)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')



# Evaluate the model
model.eval()  # Set the model to evaluation mode

with torch.no_grad():
    for inputs, labels in test_loader:  # Assume `test_loader` provides test data

        inputs = inputs.float()
        outputs = model(inputs)
        predictions = (outputs > 0.5).float()  # Convert probabilities to binary predictions (0 or 1)
        print(f"predictions:\n{predictions}\ninputs:\n{labels}")
        # Calculate accuracy or other metrics
        correct = ((predictions==1)&(labels==1)).sum().item()
        print(f"number correct {correct}")
        total_ones_in_label = (labels==1).sum().item()
        total_ones_predicted = predictions.sum().item()
        false_positives = ((predictions == 1)&(labels==0)).sum().item()
        #the math below is wrong
        accuracy = correct/total_ones_in_label if total_ones_in_label > 0 else 0.0
        print(f'Accuracy: {correct/total_ones_predicted:.4f}\nfalse positives: {false_positives}\naccuracy/false_positive: {(correct/total_ones_predicted)/false_positives}')


#OUTPUT
predictions:
tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
         0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
         0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
         0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
         0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
         0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
         0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
         0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
         0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
         0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
         0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]])
inputs:
tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
         0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]])
number correct 5
Accuracy: 0.2500
false positives: 15
accuracy/false_positive: 0.016666666666666666
