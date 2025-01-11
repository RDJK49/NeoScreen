from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def calculate_metrics(labels, predictions, threshold=0.5):
    """
    Calculate a comprehensive set of evaluation metrics.

    Args:
        labels (list or np.array): Ground truth binary labels.
        predictions (list or np.array): Predicted probabilities or scores.
        threshold (float): Threshold to convert probabilities to binary predictions.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    try:
        # Convert probabilities to binary predictions
        binary_predictions = (np.array(predictions) >= threshold).astype(int)

        # Calculate metrics
        metrics = {
            "Accuracy": accuracy_score(labels, binary_predictions),
            "ROC-AUC": roc_auc_score(labels, predictions),
            "Precision": precision_score(labels, binary_predictions),
            "Recall": recall_score(labels, binary_predictions),
            "F1-Score": f1_score(labels, binary_predictions)
        }

        return metrics

    except Exception as e:
        print(f"Error calculating metrics: {e}")
        raise

# Example usage
def evaluate_model_performance(model, dataloader, device):
    """
    Evaluate model performance on a given dataloader.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        dataloader (torch.utils.data.DataLoader): Dataloader for the evaluation dataset.
        device (torch.device): Device to run the evaluation on.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    try:
        model.eval()
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for batch in dataloader:
                smiles_input, graph_data, labels = batch
                smiles_input = smiles_input.to(device)
                labels = labels.to(device)

                graph_data.x = graph_data.x.to(device)
                graph_data.edge_index = graph_data.edge_index.to(device)
                graph_data.batch = graph_data.batch.to(device)

                outputs = model(smiles_input, graph_data)
                predictions = torch.sigmoid(outputs).cpu().numpy()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions)

        # Calculate metrics
        metrics = calculate_metrics(all_labels, all_predictions)
        return metrics

    except Exception as e:
        print(f"Error evaluating model performance: {e}")
        raise

# Review comments:
# 1. Add support for multi-class metrics if required in the future.
# 2. Ensure compatibility with both binary and multi-label tasks.
# 3. Consider logging metrics for better traceability in production.
# 4. Add unit tests to validate metrics calculation.
