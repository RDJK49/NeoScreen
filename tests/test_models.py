import unittest
import torch
from torch_geometric.data import Data, DataLoader
from models import initialize_mol_fusion
from evaluation.metrics import calculate_metrics

class TestMolFusionModel(unittest.TestCase):

    def setUp(self):
        """Set up test environment."""
        # Initialize model with sample configuration
        self.model = initialize_mol_fusion(input_dim_smiles=512, num_node_features=128, hidden_dim=256, num_classes=1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Create sample data for testing
        smiles_input = torch.rand((10, 512))  # Batch of 10 SMILES embeddings
        node_features = torch.rand((50, 128))  # 50 nodes with 128 features each
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)  # Sample edge indices
        batch = torch.tensor([0, 0, 1, 1, 1])  # Batch assignment for graph data

        self.graph_data = Data(x=node_features, edge_index=edge_index, batch=batch)
        self.smiles_input = smiles_input

        self.labels = torch.randint(0, 2, (10, 1), dtype=torch.float)  # Random binary labels

    def test_model_forward(self):
        """Test the forward pass of the model."""
        self.model.eval()

        with torch.no_grad():
            output = self.model(self.smiles_input.to(self.device), self.graph_data.to(self.device))

        self.assertEqual(output.shape, (10, 1))  # Should match batch size and output classes
        print("Forward pass test passed.")

    def test_metrics_calculation(self):
        """Test the metrics calculation function."""
        predictions = torch.sigmoid(torch.rand((10, 1))).numpy()  # Random predictions
        labels = self.labels.numpy()

        metrics = calculate_metrics(labels, predictions)
        self.assertIn("Accuracy", metrics)
        self.assertIn("ROC-AUC", metrics)
        self.assertIn("Precision", metrics)
        self.assertIn("Recall", metrics)
        self.assertIn("F1-Score", metrics)

        print("Metrics calculation test passed.")

if __name__ == "__main__":
    unittest.main()
