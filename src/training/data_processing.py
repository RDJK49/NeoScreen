import os
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split

# Utility functions for preprocessing
def smiles_to_graph(smiles):
    """
    Converts a SMILES string to a PyTorch Geometric graph.

    Args:
        smiles (str): SMILES string of the molecule.

    Returns:
        torch_geometric.data.Data: Graph representation of the molecule.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")

        # Nodes and edges
        nodes = []
        edges = []
        edge_attrs = []

        for atom in mol.GetAtoms():
            nodes.append(atom.GetAtomicNum())

        for bond in mol.GetBonds():
            edges.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            edge_attrs.append(bond.GetBondTypeAsDouble())

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        node_features = torch.tensor(nodes, dtype=torch.float).view(-1, 1)
        edge_features = torch.tensor(edge_attrs, dtype=torch.float).view(-1, 1)

        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)

    except Exception as e:
        print(f"Error converting SMILES to graph: {e}")
        return None

# Dataset preparation
def prepare_training_data(data_path, batch_size):
    """
    Prepares training, validation, and test DataLoaders.

    Args:
        data_path (str): Path to the preprocessed data file.
        batch_size (int): Batch size for DataLoaders.

    Returns:
        tuple: DataLoaders for training, validation, and test sets.
    """
    try:
        # Load dataset
        data_file = os.path.join(data_path, "molecule_data.csv")
        dataset = pd.read_csv(data_file)

        # Convert SMILES to graphs
        graphs = []
        labels = []
        for _, row in dataset.iterrows():
            graph = smiles_to_graph(row["SMILES"])
            if graph is not None:
                graph.y = torch.tensor([row["Label"]], dtype=torch.float)
                graphs.append(graph)

        # Train-test split
        train_data, test_data = train_test_split(graphs, test_size=0.2, random_state=42)
        train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

        # DataLoaders
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    except Exception as e:
        print(f"Error preparing training data: {e}")
        raise

# Example usage
if __name__ == "__main__":
    data_path = "data/processed"
    batch_size = 32
    train_loader, val_loader, test_loader = prepare_training_data(data_path, batch_size)
    print(f"Train batches: {len(train_loader)} | Validation batches: {len(val_loader)} | Test batches: {len(test_loader)}")
