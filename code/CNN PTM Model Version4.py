import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchmetrics
class PhosphoCNN(nn.Module):
    def __init__(self, sequence_length):
        super(PhosphoCNN, self).__init__()

        # First 1D convolutional layer: input channels = 21 (one-hot encoding size for each amino acid),
        # output channels = 64 (number of filters), kernel size = 3 (size of the convolutional window)
        self.conv1 = nn.Conv1d(in_channels=20, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # Second 1D convolutional layer: output channels = 128
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=11, stride=1, padding=5)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        # Max pooling layer to downsample the sequence
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Fully connected layer
        self.fc1 = nn.Linear(1024 , 128)  # Adjust size depending on the sequence length
        self.fc2 = nn.Linear(128, 64)  # Output one prediction per amino acid position
        self.fc3 = nn.Linear(64, 33)
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
        x = self.fc3(x)  # Sigmoid activation for binary classification (phosphorylation or not)
        
        return x


# Example: Train the model

# Define the model
sequence_length = 33 # Example sequence length
model = PhosphoCNN(sequence_length)

# Loss function (Binary Cross-Entropy)
# Calculate class weights
class_weights = torch.tensor([9])  # Give 10 times more weight to '1' class
#criterion = nn.BCELoss(pos_weight=class_weights)
criterion = nn.BCEWithLogitsLoss(pos_weight = class_weights)

# Optimizer (Adam)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#Torch Metrics
tval = 0.5
precision = torchmetrics.classification.BinaryPrecision(threshold = tval)
recall = torchmetrics.classification.BinaryRecall(threshold = tval)
f1 = torchmetrics.classification.BinaryF1Score(threshold = tval)
auroc = torchmetrics.classification.BinaryAUROC()
accuracy = torchmetrics.classification.Accuracy(task='binary',threshold = tval)

precision_data = []
recall_data = []
f1_data = []
auroc_data = []
accuracy_data = []
epoch_data = []

epochs = 400
# Example training loop
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    total_loss = 0
    num_loss_updates = 0
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

        total_loss += loss.item()
        num_loss_updates += 1
        
    average_loss = total_loss / num_loss_updates
    print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {average_loss:.4f}')
    
    # Validation
    model.eval()
    precision.reset()
    recall.reset()
    f1.reset()
    auroc.reset()
    with torch.no_grad():
        for inputs, labels in test_loader:  # Assume `test_loader` provides test data
            inputs = inputs.squeeze(1)  # Remove the second dimension of size 1
            inputs = inputs.permute(0, 2, 1)  # Change the order of dimensions
            inputs = inputs.float()
            #print(inputs.shape,labels.shape)
            # Forward pass
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)
            preds = (outputs > 0.5).int()
            outputs = outputs.detach()
            precision.update(outputs, labels)
            recall.update(outputs, labels)
            f1.update(outputs, labels)
            auroc.update(outputs, labels)
            accuracy.update(preds, labels)
    print(f"→ Validation Accuracy:  {accuracy.compute():.4f}")
    print(f"→ Validation Precision: {precision.compute():.4f}")
    print(f"→ Validation Recall:    {recall.compute():.4f}")
    print(f"→ Validation F1 Score:  {f1.compute():.4f}")
    print(f"→ Validation AUROC:     {auroc.compute():.4f}")

    epoch_data.append(epoch+1)
    accuracy_data.append(accuracy.compute().item())
    precision_data.append(precision.compute().item())
    recall_data.append(recall.compute().item())
    f1_data.append(f1.compute().item())
    auroc_data.append(auroc.compute().item())

precision_data = [round(value,4) for value in precision_data]
recall_data = [round(value,4) for value in recall_data]
f1_data = [round(value,4) for value in f1_data]
auroc_data = [round(value,4) for value in auroc_data]
accuracy_data = [round(value,4) for value in accuracy_data]

print()
print(f"tval = {tval}, classweight = {class_weights}, trainbatch = 100, testbatch=50, dropout=.45")
print(f"1861 Training Samples \n209 Testing Samples \nData:\n")
print(f"Epoch: {epoch_data}")
print(f"Accuracy: {accuracy_data}")
print(f"Precision: {precision_data}")
print(f"Recall: {recall_data}")
print(f"F1 Score: {f1_data}")
print(f"AUROC: {auroc_data}")


#OUTPUT WITH  TRAIN_BATCH_SIZE=100 AND TEST_BATCH_SIZE=50 AND 2K TEST DATA
#TESTED CLASSWEIGHTS FROM 14 TO 5 AND FOUND 7 TO HAVE BEST SCORE PARTICULARLY F1
Epoch [400/400], Average Loss: 0.3121
→ Validation Accuracy:  0.7718
→ Validation Precision: 0.5030
→ Validation Recall:    0.4147
→ Validation F1 Score:  0.4546
→ Validation AUROC:     0.8381

tval = 0.5, classweight = tensor([7]), trainbatch = 100, testbatch=50, dropout=.45
1861 Training Samples 
209 Testing Samples 

Epoch [400/400], Average Loss: 0.0972
→ Validation Accuracy:  0.8909
→ Validation Precision: 0.6431
→ Validation Recall:    0.7386
→ Validation F1 Score:  0.6876
→ Validation AUROC:     0.9465

tval = 0.5, classweight = tensor([7]), trainbatch = 100, testbatch=50, dropout=.45
18551 Training Samples 
4415 Testing Samples 
