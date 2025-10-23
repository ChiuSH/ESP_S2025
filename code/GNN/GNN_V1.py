import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from Bio import PDB
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 1) Parse the PDB file and extract sequence
pdb_filename = '1qk1.pdb'  # Replace with your actual PDB filename
parser = PDB.PDBParser(QUIET=True)
structure = parser.get_structure('protein', pdb_filename)

# Define a list of standard amino acids
standard_aa = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
               'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR']

# Extract sequence
sequence = []
for model in structure:
    for chain in model:
        for residue in chain:
            if residue.get_resname() in standard_aa:
                sequence.append(residue.get_resname())

print(f"Sequence length: {len(sequence)}")
print(f"Sequence: {sequence}")

# 2) Encode sequence with one-hot encoding for amino acids
aa_types = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
aa_encoder = LabelEncoder()
aa_encoder.fit(aa_types)

# Map 3-letter codes from PDB to 1-letter codes for encoding
residue_map = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 
    'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 
    'TRP': 'W', 'TYR': 'Y'
}

# One-hot encode each amino acid in the sequence
node_features = np.array([np.eye(len(aa_types))[aa_encoder.transform([residue_map[residue]])[0]] 
                          for residue in sequence])

# 3) Create edges between consecutive residues in the sequence
edges = [(i, i + 1) for i in range(len(sequence) - 1)]  # Connect consecutive residues
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # 2xE tensor

# 4) Create labels for PTM prediction (random for illustration)
# For simplicity, let's randomly assign PTM sites (0 or 1) to the residues
# In practice, you'll want to replace this with real PTM site data.
labels = np.random.randint(0, 2, size=len(sequence))  # Binary labels (0 = no PTM, 1 = PTM site)

# 5) Prepare the data for PyTorch Geometric
node_features_tensor = torch.tensor(node_features, dtype=torch.float)
edge_index_tensor = edge_index
labels_tensor = torch.tensor(labels, dtype=torch.float).view(-1, 1)  # Make it column vector

# Create a Data object
data = Data(x=node_features_tensor, edge_index=edge_index_tensor, y=labels_tensor)

# 6) Define the GNN Model
class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        return x

# 7) Set up the model, optimizer, and loss function
input_dim = len(aa_types)  # One-hot encoding length of amino acids
hidden_dim = 64           # Hidden layer dimension
output_dim = 1            # Binary classification (PTM site or not)

model = GNNModel(input_dim, hidden_dim, output_dim)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy loss for binary classification

# 8) Training Loop
epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    output = model(data)

    # Compute the loss
    loss = criterion(output, data.y)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# 9) Evaluate the Model (e.g., using accuracy or AUC)
model.eval()
with torch.no_grad():
    output = model(data)
    pred = torch.sigmoid(output).squeeze()  # Sigmoid for binary output
    predicted_labels = (pred > 0.5).float()

# Calculate accuracy or other metrics (just for demonstration)
accuracy = (predicted_labels == data.y.squeeze()).float().mean()
print(f"Accuracy: {accuracy:.4f}")
