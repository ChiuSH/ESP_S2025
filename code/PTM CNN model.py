#creating test data to see if code works
import csv
with open("code_test_data.csv",'a') as file:
    writer = csv.writer(file)
    writer.writerow(["sequence id", "protein sequence", "ptm label"])
    writer.writerow(["1", "MLTRKPSA", "0,0,0,1,0,0,0,0"])







#create dataLoader to load the test data and encode sequence
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class ProteinDataset(Dataset):
    def __init__(self, csv_file, sequence_length=5):
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
        
        # Convert the sequence into a list of amino acid indices
        amino_acid_dict = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'P': 5, 'M': 6, 'V': 7, 'K': 8, 'Q': 9, 'X': 10, 'S': 11, 'L': 12, 'R': 13, 'F': 14, 'W': 15, 'Y': 16, 'H': 17, 'I': 18, 'N': 19, 'D': 20, 'E': 21}
        
        # Convert sequence to indices
        sequence_indices = [amino_acid_dict[aa] for aa in sequence]
        
        # Padding if the sequence is shorter than the expected length
        if len(sequence_indices) < self.sequence_length:
            sequence_indices += [0] * (self.sequence_length - len(sequence_indices))  # Use 0 as padding
        
        # Convert modification labels to a list of integers
        labels = [int(label) for label in labels.split(',')]
        
        # Ensure the label length matches the sequence length
        if len(labels) < self.sequence_length:
            labels += [0] * (self.sequence_length - len(labels))  # Padding labels if necessary

        # Convert sequence and labels to tensors
        sequence_tensor = torch.tensor(sequence_indices).long()  # Long tensor for sequence indices
        labels_tensor = torch.tensor(labels).float()  # Float tensor for labels (1 or 0)

        # Reshape sequence to match the input shape for the CNN (channels, sequence length)
        sequence_tensor = sequence_tensor.unsqueeze(0)  # Add a channel dimension (1 channel for 21 amino acids)

        return sequence_tensor, labels_tensor

# Load the data into a DataLoader
dataset = ProteinDataset(csv_file="code_test_data.csv", sequence_length=8)  # Adjust sequence length based on your data
train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Example: Inspecting the loaded data
for batch_idx, (sequences, labels) in enumerate(dataloader):
    print(f"Batch {batch_idx+1}:")
    print("Sequences:", sequences)
    print("Labels:", labels)

#Output from example:
#Batch 1:
#Sequences: tensor([[[ 6, 12,  4, 13,  8,  5, 11,  1]]])
#Labels: tensor([[0., 0., 0., 1., 0., 0., 0., 0.]])









#CNN Model

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
        self.fc1 = nn.Linear(128*2 , 512)  # Adjust size depending on the sequence length
        self.fc2 = nn.Linear(512, 8)  # Output one prediction per amino acid position
        self.out = nn.Linear(8, 1)

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
sequence_length = 100  # Example sequence length
model = PhosphoCNN(sequence_length)

# Loss function (Binary Cross-Entropy)
criterion = nn.BCELoss()

# Optimizer (Adam)
optimizer = optim.Adam(model.parameters(), lr=0.001)

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
        
        outputs = model(inputs)
        predictions = (outputs > 0.5).float()  # Convert probabilities to binary predictions (0 or 1)
        
        # Calculate accuracy or other metrics
        correct = (predictions == labels).sum().item()
        accuracy = correct / labels.size(0)
        print(f'Accuracy: {accuracy:.4f}')

#Output:
#Epoch [1/10], Loss: 0.7448
#Epoch [2/10], Loss: 0.3518
#Epoch [3/10], Loss: 0.1581
#Epoch [4/10], Loss: 0.0626
#Epoch [5/10], Loss: 0.0211
#Epoch [6/10], Loss: 0.0063
#Epoch [7/10], Loss: 0.0018
#Epoch [8/10], Loss: 0.0005
#Epoch [9/10], Loss: 0.0001
#Epoch [10/10], Loss: 0.0000


