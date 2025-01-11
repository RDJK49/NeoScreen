import os
import torch
from src.models import initialize_mol_fusion
from src.training.data_processing import prepare_training_data
from src.evaluation.metrics import calculate_metrics
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

# Configuration settings
CONFIG = {
    "data_path": "data/processed/",
    "batch_size": 32,
    "input_dim_smiles": 512,
    "num_node_features": 128,
    "hidden_dim": 256,
    "num_classes": 1,
    "learning_rate": 0.001,
    "epochs": 10,
    "save_path": "result_files/best_model.pth",
}

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        smiles_input, graph_data, labels = batch
        smiles_input, labels = smiles_input.to(device), labels.to(device)

        graph_data.x = graph_data.x.to(device)
        graph_data.edge_index = graph_data.edge_index.to(device)
        graph_data.batch = graph_data.batch.to(device)

        optimizer.zero_grad()
        outputs = model(smiles_input, graph_data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in dataloader:
            smiles_input, graph_data, labels = batch
            smiles_input, labels = smiles_input.to(device), labels.to(device)

            graph_data.x = graph_data.x.to(device)
            graph_data.edge_index = graph_data.edge_index.to(device)
            graph_data.batch = graph_data.batch.to(device)

            outputs = model(smiles_input, graph_data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(torch.sigmoid(outputs).cpu().numpy())

    metrics = calculate_metrics(all_labels, all_predictions)
    return total_loss / len(dataloader), metrics

def run_pipeline(config):
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

    # Loss function and optimizer
    criterion = BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=config["learning_rate"])

    best_val_auc = 0.0

    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch + 1}/{config['epochs']}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_metrics['ROC-AUC']:.4f}")

        # Save the best model
        if val_metrics['ROC-AUC'] > best_val_auc:
            best_val_auc = val_metrics['ROC-AUC']
            torch.save(model.state_dict(), config["save_path"])
            print("Best model saved.")

    # Test the model
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test AUC: {test_metrics['ROC-AUC']:.4f}")

if __name__ == "__main__":
    run_pipeline(CONFIG)
