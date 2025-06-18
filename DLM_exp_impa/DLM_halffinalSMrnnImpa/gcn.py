import torch
import torch_geometric
from torch_geometric.nn import GCNConv
from torch import nn

# speaker encoder (meaning - hidden state) (hidden state - utterance)
# listener decoder (utterance - hidden state) (hidden state - meaning)

# VERB_7 ANIMATE_3 ADPOSITION_3 ADJECTIVE_3 INANIMATE_1 ANIMATE_8
# Action Agent Preposition_clause1 Preposition_clause2 Prepositional_clause3 Patient
# edge_index = torch.tensor([
#     [0, 1],  # Action -> Agent
#     [1, 2],  # Agent -> Preposition_clause1
#     [2, 3],  # Preposition_clause1 -> Preposition_clause2
#     [3, 4],  # Preposition_clause2 -> Prepositional_clause3

edge_index = torch.tensor([
    [0, 1],  # I -> like (subject-verb)
    [1, 2],  # like -> cat (verb-object)
    [2, 3],  # cat -> with (noun-preposition)
    [3, 5],  # with -> fur (preposition-noun)
    [5, 4]   # fur -> white (adjective-noun)
], dtype=torch.long).t().contiguous()


############## meaning to hidden state ##############
class SentenceGNN(nn.Module):
    def __init__(self, n_hidden, n_features):
        super(SentenceGNN, self).__init__()
        self.gcn = GCNConv(n_features, n_hidden)

    def forward(self, x, edge_index):
        # x: [num_nodes, num_features] (node features)
        # edge_index: [2, num_edges] (graph connectivity)
        x = self.gcn(x, edge_index)
        return x

# Define number of nodes and features
n_features = 6  # Example feature size for each meaning slot
n_hidden = 8  # Hidden state size

# Initialize the model
model = SentenceGNN(n_hidden=n_hidden, n_features=n_features)

# Example feature vectors for the 6 meaning slots
node_features = torch.rand((6, n_features))  # Random initialization for now

# Define edges (dependencies between meaning slots)
# Edge index for the graph as described earlier
edge_index = torch.tensor([
    [0, 1],  # I -> have
    [1, 2],  # have -> cat
    [2, 3],  # cat -> with
    [3, 5],  # with -> fur
    [5, 4]   # fur -> white
], dtype=torch.long).t().contiguous()

# Forward pass through GNN
output = model(node_features, edge_index)
print(output)


import torch
from torch_geometric.nn import GCNConv
from torch import nn

# Step 1: Define the GCN model (from your example)
class SentenceGNN(nn.Module):
    def __init__(self, n_hidden, n_features):
        super(SentenceGNN, self).__init__()
        self.gcn = GCNConv(n_features, n_hidden)

    def forward(self, x, edge_index):
        x = self.gcn(x, edge_index)
        return x

# Step 2: Define a small graph with 5 meaning slots (representing "I have a white cat")
n_features = 4  # Each meaning slot will have 4 features
n_hidden = 6    # Hidden state size after GCN layer

# Initialize the GCN model
model = SentenceGNN(n_hidden=n_hidden, n_features=n_features)

# Step 3: Define the input features for each meaning slot
# Random feature vectors for the words in "I have a white cat"
node_features = torch.tensor([
    [0.2, 0.4, 0.6, 0.8],  # I (meaning slot 1)
    [0.3, 0.7, 0.5, 0.2],  # have (meaning slot 2)
    [0.5, 0.1, 0.3, 0.7],  # white (meaning slot 3)
    [0.6, 0.9, 0.2, 0.1],  # cat (meaning slot 4)
    [0.4, 0.5, 0.3, 0.6],  # a (meaning slot 5)
], dtype=torch.float)

# Step 4: Define the graph structure (edges between nodes)
# These edges represent dependencies between the words in "I have a white cat"
edge_index = torch.tensor([
    [0, 1],  # I -> have
    [1, 4],  # have -> a
    [4, 3],  # a -> cat
    [3, 2],  # cat -> white
], dtype=torch.long).t().contiguous()

# Step 5: Perform a forward pass through the GCN layer
output = model(node_features, edge_index)

# Step 6: Output the result
print("Node feature vectors after GCN processing:")
print(output)


################ hidden state to meaning ################
class Seq2Graph(nn.Module):
    def __init__(self, n_hidden, n_features):
        super(Seq2Graph, self).__init__()
        self.rnn_encoder = nn.LSTM(n_features, n_hidden, batch_first=True)  ### listener encoder

        # listener decoder
        self.node_predictor = nn.Linear(n_hidden, n_features)  # Predict node features
        self.edge_predictor = nn.Linear(n_hidden * 2, 1)  # Predict edge between two nodes

    def forward(self, sequence):
        # Encode the input sequence into hidden representations
        encoded_sequence, _ = self.rnn_encoder(sequence)  # [batch_size, seq_len, n_hidden]
        
        # Predict nodes
        node_features = self.node_predictor(encoded_sequence)  # [batch_size, seq_len, n_features]
        
        # Predict edges (dependencies) between nodes
        edge_scores = []
        for i in range(sequence.size(1)):
            for j in range(sequence.size(1)):
                edge_input = torch.cat([encoded_sequence[:, i, :], encoded_sequence[:, j, :]], dim=-1)
                edge_score = self.edge_predictor(edge_input)
                edge_scores.append(edge_score)

        edge_scores = torch.stack(edge_scores, dim=1)  # [batch_size, seq_len^2, 1]
        
        return node_features, edge_scores
