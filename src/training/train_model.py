import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
from models import initialize_mol_fusion
from data_processing import prepare_training_data

# Training function
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0.0
    all_labels = []
    all_preds = []

    for batch in dataloader:
        smiles_input, graph_data, labels = batch
        smiles_input = smiles_input.to(device)
        labels = labels.to(device)

        # Move graph data to device
        graph_data.x = graph_data.x.to(device)
        graph_data.edge_index = graph_data.edge_index.to(device)
        graph_data.batch = graph_data.batch.to(device)

        optimizer.zero_grad()
        outputs = model(smiles_input, graph_data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())

    epoch_loss /= len(dataloader)
    return epoch_loss, all_labels, all_preds

# Evaluation function
def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            smiles_input, graph_data, labels = batch
            smiles_input = smiles_input.to(device)
            labels = labels.to(device)

            graph_data.x = graph_data.x.to(device)
            graph_data.edge_index = graph_data.edge_index.to(device)
            graph_data.batch = graph_data.batch.to(device)

            outputs = model(smiles_input, graph_data)
            loss = criterion(outputs, labels)

            epoch_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())

    epoch_loss /= len(dataloader)
    return epoch_loss, all_labels, all_preds

# Main training function
def main_training(config):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Prepare data
        train_loader, val_loader, test_loader = prepare_training_data(config["data_path"], config["batch_size"])

        # Initialize model
        model = initialize_mol_fusion(
            input_dim_smiles=config["input_dim_smiles"],
            num_node_features=config["num_node_features"],
            hidden_dim=config["hidden_dim"],
            num_classes=config["num_classes"]
        ).to(device)

        criterion = nn.BCEWithLogitsLoss() if config["num_classes"] == 1 else nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

        best_val_auc = 0.0

        for epoch in range(config["epochs"]):
            print(f"Epoch {epoch + 1}/{config['epochs']}")

            train_loss, train_labels, train_preds = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_labels, val_preds = evaluate(model, val_loader, criterion, device)

            train_auc = roc_auc_score(train_labels, train_preds)
            val_auc = roc_auc_score(val_labels, val_preds)

            print(f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")

            # Save the best model
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), config["save_path"])
                print("Best model saved.")

        # Test the model
        test_loss, test_labels, test_preds = evaluate(model, test_loader, criterion, device)
        test_auc = roc_auc_score(test_labels, test_preds)
        print(f"Test Loss: {test_loss:.4f} | Test AUC: {test_auc:.4f}")

    except Exception as e:
        print(f"Error in training: {e}")
        raise

# Configuration for training
config = {
    "data_path": "data/processed/",
    "batch_size": 32,
    "input_dim_smiles": 512,
    "num_node_features": 128,
    "hidden_dim": 256,
    "num_classes": 1,
    "learning_rate": 0.001,
    "epochs": 20,
    "save_path": "models/mol_fusion_best.pth"
}

if __name__ == "__main__":
    main_training(config)
