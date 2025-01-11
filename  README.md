# NeoScreen
**Author**: Rishidharan Jayakumar  
**Affiliation**: South Iredell High School, Troutman, NC


NeoScreen is a hybrid deep learning framework for virtual screening and target prediction in drug discovery. Combining ligand-based, structure-based, and graph-based approaches, NeoScreen leverages state-of-the-art techniques to achieve high accuracy and scalability.

---

## **Project Structure**

```
project/
├── data/                    # Data-related files
│   ├── descriptors/         # Molecular descriptors (e.g., RDKit features)
│   ├── processed/           # Processed datasets
│   ├── raw/                 # Raw input data
├── notebooks/               # Jupyter notebooks for experimentation
├── result_files/            # Model outputs and results
├── src/                     # Source code
│   ├── evaluation/          # Evaluation metrics and functions
│   │   ├── __init__.py
│   │   ├── metrics.py
│   ├── models/              # Model architectures
│   │   ├── __init__.py
│   │   ├── models.py
│   ├── training/            # Training-related scripts
│   │   ├── __init__.py
│   │   ├── data_processing.py
│   │   ├── train_model.py
│   ├── utils/               # Utility scripts
│   │   ├── __init__.py
│   │   ├── evaluation_metrics.py
│   │   ├── data_processing.py
├── tests/                   # Test cases for models and training
├── README.md                # Project documentation
├── run_pipeline.py          # Main pipeline script to run the framework
├── .env                     # Environment variables
├── setup.py                 # Package setup file
```

---

## **Features**

- **Hybrid Modeling**:
  - Combines SMILES-based transformer encoders and graph-based neural networks (GCN).
- **Comprehensive Data Handling**:
  - Preprocessing tools for SMILES, molecular graphs, and descriptors.
- **Advanced Metrics**:
  - Evaluates models using accuracy, ROC-AUC, precision, recall, and F1-score.
- **End-to-End Pipeline**:
  - Supports training, evaluation, and testing with checkpointing and logging.

---

## **Installation**

### Prerequisites

- Python 3.8+
- PyTorch and PyTorch Geometric
- RDKit for molecular preprocessing

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## **Usage**

### **Run the Pipeline**

Use the `run_pipeline.py` script to train and evaluate the model.

```bash
python run_pipeline.py
```

### **Configuration**

Edit the `CONFIG` dictionary in `run_pipeline.py` to customize the training parameters:

```python
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
```

### **Run Tests**

Run unit tests to verify the framework:

```bash
python -m unittest discover tests
```

---

## **Pipeline Details**

1. **Data Preparation**:
   - Converts SMILES strings to molecular graphs.
   - Splits data into training, validation, and test sets.

2. **Model**:
   - `MolFusion`: Combines SMILES and graph-based representations.

3. **Training**:
   - Trains the model with checkpointing and logs validation metrics.

4. **Evaluation**:
   - Computes metrics like accuracy, ROC-AUC, precision, recall, and F1-score.

---

## **Folder Details**

- `data/`: Contains raw and processed datasets.
- `src/models/`: Model definitions, including `MolFusion`.
- `src/training/`: Scripts for data preprocessing and training.
- `src/evaluation/`: Evaluation metric functions.
- `tests/`: Unit tests for the framework.

---

## **Contributing**

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`.
3. Commit your changes: `git commit -m "Add feature"`.
4. Push to the branch: `git push origin feature-name`.
5. Open a pull request.

---

## **License**

This project is licensed under the MIT License. See the LICENSE file for details.

---

## **Acknowledgments**

This project was created by **Rishidharan Jayakumar**, a student at South Iredell High School, Troutman, NC.


- DEEPScreen: [GitHub Repo](https://github.com/cansyl/DEEPScreen)
- DeepVS: [GitHub Repo](https://github.com/JanainaCruz/DeepVS)
- DeepLBVS: [GitHub Repo](https://github.com/taneishi/DeepLBVS)
- PyRMD: [GitHub Repo](https://github.com/cosconatilab/PyRMD)
- Supported by RDKit and PyTorch Geometric.

