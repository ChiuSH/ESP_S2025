#DATA LOADER
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
#import numpy as np
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




    def random_shift_sequence(self,sequence, label): #does nothing for now
        """Randomly shifts a sequence and adjusts the label accordingly."""
        shift = np.random.randint(-3, 4)  # Shift left or right by up to 3 positions
        sequence = np.roll(sequence, shift, axis=0)  # Shift sequence
        label = np.roll(label, shift, axis=0)  # Shift labels accordingly
        return sequence, label

    def __getitem__(self, idx):
        # Get the protein sequence and modification labels
        sequence = self.data.iloc[idx, 1]  # Protein sequence column
        labels = self.data.iloc[idx, 2]  # Modification labels column
        #sequence = np.array(list(sequence))  # Convert string sequence into array <--added
        #labels = np.array([int(x) for x in labels.split(',')])#<-- added
        #sequence, label = self.random_shift_sequence(sequence, labels)#<-- added
        #sequence = str(sequence)
        #labels = str(labels)
        sequence = sequence.replace('\r','').replace('\n','').replace('"','').replace("'",'').replace(" ",'').replace('[','').replace(']','')
        labels = labels.replace('\r','').replace('\n','').replace('"','').replace("'",'').replace(" ",'').replace('[','').replace(']','')[:]
        #print(f"label before processing {labels} and length: {len(labels)}") --------------------------- helps with bug fixing
        #print(f"sequence before processing {sequence} and length: {len(sequence)}") --------------------
        # Convert the sequence into a list of amino acid indices
        amino_acid_dict = {'A': [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'R': [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'N': [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'D': [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'C': [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'E': [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'Q': [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0], 'G': [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0], 'H': [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0], 'I': [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0], 'L': [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0], 'K': [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0], 'M': [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0], 'F': [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0], 'P': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0], 'S': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0], 'T': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0], 'W': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0], 'Y': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0], 'V': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]}
        
        # Convert sequence to indices
        sequence_indices = [amino_acid_dict[aa] for aa in sequence]
        
        # Padding if the sequence is shorter than the expected length
        #if len(sequence_indices) < self.sequence_length:
            #sequence_indices += [0] * (self.sequence_length - len(sequence_indices))  # Use 0 as padding
        #print(f"sequence indices is: {sequence_indices} and length: {len(sequence_indices)}")  -----------
        # Convert modification labels to a list of integers
        labels = [int(label) for label in labels.split(",")]
        
        # Ensure the label length matches the sequence length
        #if len(labels) < self.sequence_length:
            #labels += [0] * (self.sequence_length - len(labels))  # Padding labels if necessary
        #print(f"label indices is: {labels} and length: {len(labels)}")-------------------------------------
        # Convert sequence and labels to tensors
        sequence_tensor = torch.tensor(sequence_indices).long()  # Long tensor for sequence indices
        labels_tensor = torch.tensor(labels).float()  # Float tensor for labels (1 or 0)

        # Reshape sequence to match the input shape for the CNN (channels, sequence length)
        sequence_tensor = sequence_tensor.unsqueeze(0)  # Add a channel dimension (1 channel for 21 amino acids)

        return sequence_tensor, labels_tensor

# Load the data into a DataLoader
dataset = ProteinDataset(csv_file="code_test_data.csv", sequence_length=33)  # Adjust sequence length based on your data
train_loader = DataLoader(dataset, batch_size=200, shuffle=True)
data_pred = ProteinDataset(csv_file="code_predict.csv", sequence_length=33)
test_loader = DataLoader(data_pred, batch_size=100, shuffle=True)
# Example: Inspecting the loaded data
for batch_idx, (sequences, labels) in enumerate(train_loader):
    print(f"Batch {batch_idx+1}:")
    print("Sequences:", sequences)
    print("Labels:", labels)


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#CNN MODEL

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class PhosphoCNN(nn.Module):
    def __init__(self, sequence_length):
        super(PhosphoCNN, self).__init__()

        # First 1D convolutional layer: input channels = 21 (one-hot encoding size for each amino acid),
        # output channels = 64 (number of filters), kernel size = 3 (size of the convolutional window)
        self.conv1 = nn.Conv1d(in_channels=20, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # Second 1D convolutional layer: output channels = 128
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=11, stride=1, padding=5)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0)
        # Max pooling layer to downsample the sequence
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Fully connected layer
        self.fc1 = nn.Linear(768 , 256)  # Adjust size depending on the sequence length
        self.fc2 = nn.Linear(256, 128)  # Output one prediction per amino acid position
        self.fc3 = nn.Linear(128, 33)
        self.out = nn.Linear(33, 1)
        self.dropout = nn.Dropout(p=0.45)

    def forward(self, x):
        # Apply first convolution, ReLU activation, and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        
        # Apply second convolution, ReLU activation, and max pooling
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        # Flatten the output from 2D to 1D (for fully connected layer)
        x = x.view(-1, 256 * (x.size(2)))  # Adjust the size based on sequence length
        
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Sigmoid activation for binary classification (phosphorylation or not)
        
        return x


# Example: Train the model

# Define the model
sequence_length = 33 # Example sequence length
model = PhosphoCNN(sequence_length)

# Loss function (Binary Cross-Entropy)
# Calculate class weights
class_weights = torch.tensor([14])  # Give 10 times more weight to '1' class
#criterion = nn.BCELoss(pos_weight=class_weights)
criterion = nn.BCEWithLogitsLoss(pos_weight = class_weights)

# Optimizer (Adam)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

epochs = 200
# Example training loop
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    
    for inputs, labels in train_loader:  # Assume `train_loader` provides protein sequences and labels
        optimizer.zero_grad()
        inputs=inputs.float()
        #fix dimension of input with matrix
        # Assuming your input shape is [111, 1, 33, 20]
# We need to remove the unnecessary dimension 1 and transpose the tensor to [batch_size, channels, sequence_length]
        inputs = inputs.squeeze(1)  # Remove the second dimension of size 1
        inputs = inputs.permute(0, 2, 1)  # Change shape from [batch_size, sequence_length, channels] to [batch_size, channels, sequence_length]

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

total_correct = 0
total_ones_in_label = 0
total_ones_predicted = 0
total_false_positives = 0
total_samples = 0  # To track the total number of samples

with torch.no_grad():
    for inputs, labels in test_loader:  # Assume `test_loader` provides test data
        inputs = inputs.squeeze(1)  # Remove the second dimension of size 1
        inputs = inputs.permute(0, 2, 1)  # Change the order of dimensions
        inputs = inputs.float()
        
        # Forward pass
        outputs = model(inputs)
        
        # Convert outputs to binary predictions (0 or 1)
        predictions = (outputs > 0.5).float()
        print(f"predictions:\n{predictions}\ninputs:\n{labels}\n")
        # Calculate number of correct predictions
        correct = ((predictions == 1) & (labels == 1)).sum().item()
        total_correct += correct
        
        # Calculate total number of ones in labels and predictions
        total_ones_in_label += (labels == 1).sum().item()
        total_ones_predicted += predictions.sum().item()
        
        # Calculate false positives
        false_positives = ((predictions == 1) & (labels == 0)).sum().item()
        total_false_positives += false_positives
        
        total_samples += labels.size(0)  # Add the number of samples in this batch

# Calculate final accuracy across all batches
accuracy = total_correct / total_ones_in_label if total_ones_in_label > 0 else 0.0
false_positive_rate = total_false_positives / total_samples if total_samples > 0 else 0.0

print(f"Final Accuracy: {accuracy}")
print(f"Total False Positives: {total_false_positives}")
print(f"False Positive Rate: {false_positive_rate}")



#OUTPUT

Epoch [1/200], Loss: 1.7057
Epoch [2/200], Loss: 1.6751
Epoch [3/200], Loss: 1.6623
Epoch [4/200], Loss: 1.6528
Epoch [5/200], Loss: 1.6774
Epoch [6/200], Loss: 1.6473
Epoch [7/200], Loss: 1.6838
Epoch [8/200], Loss: 1.6969
Epoch [9/200], Loss: 1.7032
Epoch [10/200], Loss: 1.6964
Epoch [11/200], Loss: 1.6976
Epoch [12/200], Loss: 1.6822
Epoch [13/200], Loss: 1.6766
Epoch [14/200], Loss: 1.6750
Epoch [15/200], Loss: 1.6664
Epoch [16/200], Loss: 1.6478
Epoch [17/200], Loss: 1.6438
Epoch [18/200], Loss: 1.6852
Epoch [19/200], Loss: 1.6571
Epoch [20/200], Loss: 1.7007
Epoch [21/200], Loss: 1.6173
Epoch [22/200], Loss: 1.7002
Epoch [23/200], Loss: 1.6817
Epoch [24/200], Loss: 1.6473
Epoch [25/200], Loss: 1.6128
Epoch [26/200], Loss: 1.6355
Epoch [27/200], Loss: 1.6532
Epoch [28/200], Loss: 1.6684
Epoch [29/200], Loss: 1.6516
Epoch [30/200], Loss: 1.5909
Epoch [31/200], Loss: 1.6612
Epoch [32/200], Loss: 1.6438
Epoch [33/200], Loss: 1.6173
Epoch [34/200], Loss: 1.5400
Epoch [35/200], Loss: 1.5896
Epoch [36/200], Loss: 1.6079
Epoch [37/200], Loss: 1.5883
Epoch [38/200], Loss: 1.5994
Epoch [39/200], Loss: 1.5942
Epoch [40/200], Loss: 1.6172
Epoch [41/200], Loss: 1.6202
Epoch [42/200], Loss: 1.5841
Epoch [43/200], Loss: 1.6071
Epoch [44/200], Loss: 1.6052
Epoch [45/200], Loss: 1.6128
Epoch [46/200], Loss: 1.6050
Epoch [47/200], Loss: 1.5619
Epoch [48/200], Loss: 1.5549
Epoch [49/200], Loss: 1.6000
Epoch [50/200], Loss: 1.5413
Epoch [51/200], Loss: 1.6106
Epoch [52/200], Loss: 1.6159
Epoch [53/200], Loss: 1.5981
Epoch [54/200], Loss: 1.5802
Epoch [55/200], Loss: 1.6074
Epoch [56/200], Loss: 1.5699
Epoch [57/200], Loss: 1.5605
Epoch [58/200], Loss: 1.6264
Epoch [59/200], Loss: 1.6328
Epoch [60/200], Loss: 1.5697
Epoch [61/200], Loss: 1.5797
Epoch [62/200], Loss: 1.5701
Epoch [63/200], Loss: 1.5718
Epoch [64/200], Loss: 1.5737
Epoch [65/200], Loss: 1.5575
Epoch [66/200], Loss: 1.5569
Epoch [67/200], Loss: 1.5419
Epoch [68/200], Loss: 1.5674
Epoch [69/200], Loss: 1.5547
Epoch [70/200], Loss: 1.5371
Epoch [71/200], Loss: 1.5470
Epoch [72/200], Loss: 1.5705
Epoch [73/200], Loss: 1.5952
Epoch [74/200], Loss: 1.5113
Epoch [75/200], Loss: 1.5672
Epoch [76/200], Loss: 1.5344
Epoch [77/200], Loss: 1.5694
Epoch [78/200], Loss: 1.5570
Epoch [79/200], Loss: 1.5577
Epoch [80/200], Loss: 1.5266
Epoch [81/200], Loss: 1.5553
Epoch [82/200], Loss: 1.5316
Epoch [83/200], Loss: 1.5997
Epoch [84/200], Loss: 1.5648
Epoch [85/200], Loss: 1.5202
Epoch [86/200], Loss: 1.6021
Epoch [87/200], Loss: 1.5207
Epoch [88/200], Loss: 1.5216
Epoch [89/200], Loss: 1.5271
Epoch [90/200], Loss: 1.4721
Epoch [91/200], Loss: 1.5197
Epoch [92/200], Loss: 1.5102
Epoch [93/200], Loss: 1.5617
Epoch [94/200], Loss: 1.5320
Epoch [95/200], Loss: 1.5227
Epoch [96/200], Loss: 1.4975
Epoch [97/200], Loss: 1.5437
Epoch [98/200], Loss: 1.5127
Epoch [99/200], Loss: 1.5132
Epoch [100/200], Loss: 1.5202
Epoch [101/200], Loss: 1.5489
Epoch [102/200], Loss: 1.5258
Epoch [103/200], Loss: 1.5148
Epoch [104/200], Loss: 1.5407
Epoch [105/200], Loss: 1.4732
Epoch [106/200], Loss: 1.5036
Epoch [107/200], Loss: 1.4989
Epoch [108/200], Loss: 1.4888
Epoch [109/200], Loss: 1.4447
Epoch [110/200], Loss: 1.5399
Epoch [111/200], Loss: 1.4545
Epoch [112/200], Loss: 1.5349
Epoch [113/200], Loss: 1.5326
Epoch [114/200], Loss: 1.4938
Epoch [115/200], Loss: 1.5644
Epoch [116/200], Loss: 1.4702
Epoch [117/200], Loss: 1.5140
Epoch [118/200], Loss: 1.4800
Epoch [119/200], Loss: 1.4741
Epoch [120/200], Loss: 1.4594
Epoch [121/200], Loss: 1.5258
Epoch [122/200], Loss: 1.5056
Epoch [123/200], Loss: 1.5042
Epoch [124/200], Loss: 1.5032
Epoch [125/200], Loss: 1.4949
Epoch [126/200], Loss: 1.4747
Epoch [127/200], Loss: 1.4749
Epoch [128/200], Loss: 1.5470
Epoch [129/200], Loss: 1.4393
Epoch [130/200], Loss: 1.4359
Epoch [131/200], Loss: 1.4945
Epoch [132/200], Loss: 1.5190
Epoch [133/200], Loss: 1.4860
Epoch [134/200], Loss: 1.4763
Epoch [135/200], Loss: 1.4972
Epoch [136/200], Loss: 1.4580
Epoch [137/200], Loss: 1.4913
Epoch [138/200], Loss: 1.4745
Epoch [139/200], Loss: 1.4859
Epoch [140/200], Loss: 1.4614
Epoch [141/200], Loss: 1.4274
Epoch [142/200], Loss: 1.5496
Epoch [143/200], Loss: 1.5062
Epoch [144/200], Loss: 1.5140
Epoch [145/200], Loss: 1.4592
Epoch [146/200], Loss: 1.5004
Epoch [147/200], Loss: 1.4476
Epoch [148/200], Loss: 1.4506
Epoch [149/200], Loss: 1.4427
Epoch [150/200], Loss: 1.4874
Epoch [151/200], Loss: 1.4798
Epoch [152/200], Loss: 1.4685
Epoch [153/200], Loss: 1.4838
Epoch [154/200], Loss: 1.4209
Epoch [155/200], Loss: 1.5374
Epoch [156/200], Loss: 1.4847
Epoch [157/200], Loss: 1.4581
Epoch [158/200], Loss: 1.4875
Epoch [159/200], Loss: 1.5414
Epoch [160/200], Loss: 1.5149
Epoch [161/200], Loss: 1.4396
Epoch [162/200], Loss: 1.4579
Epoch [163/200], Loss: 1.4905
Epoch [164/200], Loss: 1.4979
Epoch [165/200], Loss: 1.4372
Epoch [166/200], Loss: 1.4963
Epoch [167/200], Loss: 1.4817
Epoch [168/200], Loss: 1.5395
Epoch [169/200], Loss: 1.5312
Epoch [170/200], Loss: 1.5105
Epoch [171/200], Loss: 1.4816
Epoch [172/200], Loss: 1.4592
Epoch [173/200], Loss: 1.5034
Epoch [174/200], Loss: 1.4880
Epoch [175/200], Loss: 1.4915
Epoch [176/200], Loss: 1.4793
Epoch [177/200], Loss: 1.5165
Epoch [178/200], Loss: 1.5057
Epoch [179/200], Loss: 1.4204
Epoch [180/200], Loss: 1.4707
Epoch [181/200], Loss: 1.4583
Epoch [182/200], Loss: 1.4941
Epoch [183/200], Loss: 1.4654
Epoch [184/200], Loss: 1.5139
Epoch [185/200], Loss: 1.4842
Epoch [186/200], Loss: 1.4840
Epoch [187/200], Loss: 1.5525
Epoch [188/200], Loss: 1.4813
Epoch [189/200], Loss: 1.4194
Epoch [190/200], Loss: 1.4820
Epoch [191/200], Loss: 1.5152
Epoch [192/200], Loss: 1.4784
Epoch [193/200], Loss: 1.4775
Epoch [194/200], Loss: 1.4942
Epoch [195/200], Loss: 1.4595
Epoch [196/200], Loss: 1.4271
Epoch [197/200], Loss: 1.5098
Epoch [198/200], Loss: 1.4422
Epoch [199/200], Loss: 1.4647
Epoch [200/200], Loss: 1.5137
predictions:
tensor([[0., 0., 0.,  ..., 0., 0., 1.],
        [0., 0., 0.,  ..., 0., 1., 0.],
        [1., 1., 1.,  ..., 1., 1., 0.],
        ...,
        [1., 1., 1.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 1., 1., 1.]])
inputs:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 1.,  ..., 0., 0., 0.],
        ...,
        [1., 0., 1.,  ..., 0., 0., 0.],
        [0., 0., 1.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 1.]])

predictions:
tensor([[0., 0., 0.,  ..., 1., 1., 1.],
        [1., 1., 1.,  ..., 1., 1., 1.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 1.,  ..., 1., 0., 0.],
        [0., 0., 1.,  ..., 1., 1., 0.]])
inputs:
tensor([[0., 0., 0.,  ..., 0., 1., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 1.,  ..., 1., 1., 0.],
        [0., 0., 0.,  ..., 0., 1., 1.]])

predictions:
tensor([[1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
         1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 0.],
        [1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 1., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
         1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 1., 1., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
inputs:
tensor([[0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
         0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [1., 0., 0., 1., 0., 1., 0., 0., 1., 0., 0., 0., 0., 1., 1., 1., 0., 1.,
         0., 1., 0., 1., 1., 0., 0., 0., 0., 1., 0., 1., 0., 1., 1.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
        [1., 0., 0., 1., 0., 0., 0., 0., 1., 1., 1., 0., 1., 0., 1., 0., 1., 1.,
         0., 0., 0., 0., 1., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 1., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 1., 1.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

Final Accuracy: 0.4725
Total False Positives: 1283
False Positive Rate: 6.138755980861244

