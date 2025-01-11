import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class MolFusion(nn.Module):
    def __init__(self, input_dim_smiles: int, num_node_features: int, hidden_dim: int, num_classes: int):
        """
        MolFusion: A hybrid model combining transformer-based encoding for SMILES
        and GCN-based encoding for molecular graphs.

        Args:
            input_dim_smiles (int): Dimensionality of SMILES input embeddings.
            num_node_features (int): Number of features per graph node.
            hidden_dim (int): Hidden layer size for the fusion layer.
            num_classes (int): Number of output classes.
        """
        super(MolFusion, self).__init__()

        # Transformer-based SMILES encoder
        self.smiles_encoder = nn.Sequential(
            nn.Linear(input_dim_smiles, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Graph-based GCN encoder
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, smiles_input, graph_data):
        """
        Forward pass for the MolFusion model.

        Args:
            smiles_input (torch.Tensor): Input tensor for SMILES data [batch_size, input_dim_smiles].
            graph_data (torch_geometric.data.Data): Batched graph data.

        Returns:
            torch.Tensor: Output predictions [batch_size, num_classes].
        """
        # SMILES encoding
        smiles_embedding = self.smiles_encoder(smiles_input)

        # GCN encoding
        x, edge_index, batch = graph_data.x, graph_data.edge_index, graph_data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        graph_embedding = global_mean_pool(x, batch)  # Pool over graph nodes

        # Fusion
        combined_embedding = torch.cat((smiles_embedding, graph_embedding), dim=1)
        fused_output = self.fusion_layer(combined_embedding)

        # Output
        output = self.output_layer(fused_output)
        return output

# Example instantiation
def initialize_mol_fusion(input_dim_smiles: int, num_node_features: int, hidden_dim: int, num_classes: int):
    try:
        model = MolFusion(input_dim_smiles, num_node_features, hidden_dim, num_classes)
        return model
    except Exception as e:
        raise RuntimeError(f"Error initializing MolFusion model: {e}")

# Review comments:
# 1. Ensure input dimensions (e.g., SMILES and graph data) match expected values.
# 2. Consider integrating positional encodings for SMILES inputs.
# 3. Use batch normalization for better training stability if needed.
# 4. Add regularization techniques like dropout to avoid overfitting.
