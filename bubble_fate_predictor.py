import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import os
import random
from tqdm import tqdm

try:
    import MDAnalysis as mda
except ImportError:
    print("MDAnalysis not installed. Please run: pip install MDAnalysis")
    mda = None 

# --- Configuration Parameters ---
VOXEL_GRID_SIZE = 32                  
VOXEL_CUBE_HALF_SIDE = 15.0           
BATCH_SIZE = 32
NUM_EPOCHS = 20
NUM_CLASSES = 4
CLASS_NAMES = ["Stable/Spherical", "Coalescence Imminent", "Dissolution/Collapse", "Stabilized Peanut"]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "bubble_fate_predictor_model.pth" 

# --- Utility Functions for Model Persistence ---

def save_model(model, path):
    """Saves the model's state dictionary to the specified path."""
    print(f"Saving model to {path}...")
    torch.save(model.state_dict(), path)
    print("Model saved successfully.")

def load_model(model, path, device):
    """Loads the model's state dictionary from the specified path."""
    if os.path.exists(path):
        print(f"Loading existing model from {path}...")
        model.load_state_dict(torch.load(path, map_location=device))
        print("Model loaded successfully.")
        return True
    else:
        print(f"No existing model found at {path}. Model will be trained from scratch.")
        return False
        
# --- Data Processing Functions (Unchanged) ---

def find_bubble_center(frame_coords, gas_indices, box_dims):
    if gas_indices.size == 0:
        return box_dims / 2.0
    gas_coords = frame_coords[gas_indices]
    center = np.mean(gas_coords, axis=0)
    return center

def voxelize_bubble(coords, center, box_dims):
    coords_centered = coords - center
    coords_pbc = coords_centered - np.round(coords_centered / box_dims) * box_dims
    H = VOXEL_CUBE_HALF_SIDE
    
    mask = (coords_pbc[:, 0] >= -H) & (coords_pbc[:, 0] < H) & \
           (coords_pbc[:, 1] >= -H) & (coords_pbc[:, 1] < H) & \
           (coords_pbc[:, 2] >= -H) & (coords_pbc[:, 2] < H)
           
    cropped_coords = coords_pbc[mask]
    
    if cropped_coords.size == 0:
        return np.zeros((VOXEL_GRID_SIZE, VOXEL_GRID_SIZE, VOXEL_GRID_SIZE), dtype=np.float32)

    scaling_factor = VOXEL_GRID_SIZE / (2 * H)
    voxel_indices = (cropped_coords + H) * scaling_factor
    voxel_indices = np.clip(voxel_indices, 0, VOXEL_GRID_SIZE - 1).astype(int)
    
    voxel_grid = np.zeros((VOXEL_GRID_SIZE, VOXEL_GRID_SIZE, VOXEL_GRID_SIZE), dtype=np.float32)
    
    flattened_indices = (voxel_indices[:, 0] * VOXEL_GRID_SIZE**2) + \
                        (voxel_indices[:, 1] * VOXEL_GRID_SIZE) + \
                        voxel_indices[:, 2]
    
    counts = np.bincount(flattened_indices, minlength=VOXEL_GRID_SIZE**3)
    voxel_grid = counts.reshape((VOXEL_GRID_SIZE, VOXEL_GRID_SIZE, VOXEL_GRID_SIZE)).astype(np.float32)

    max_count = np.max(voxel_grid)
    if max_count > 0:
        voxel_grid /= max_count
        
    return voxel_grid

class BubbleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float().unsqueeze(1) 
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ConvNet3D(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet3D, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        
        self.flatten_size = 64 * (VOXEL_GRID_SIZE // 4)**3
        
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, self.flatten_size) 
        x = self.fc(x)
        return x

def load_and_process_trajectories(dump_file_paths, y_labels_list, gas_atom_type=1):
    if mda is None:
        print("Cannot load data: MDAnalysis is required.")
        return load_mock_trajectories()
        
    X_ALL = []
    Y_ALL = []
    
    atom_style_str = 'id type x y z vx vy vz c_dstress[1] c_dstress[2] c_dstress[3]'
    
    if len(dump_file_paths) != len(y_labels_list):
        print("ERROR: The number of file paths does not match the number of label arrays.")
        return load_mock_trajectories()

    print(f"\n--- Loading and Processing {len(dump_file_paths)} Trajectories ---")

    for file_path, Y_LABELS in zip(dump_file_paths, y_labels_list):
        print(f"\nProcessing File: {file_path}")
        
        try:
            u = mda.Universe(file_path, format='LAMMPS', atom_style=atom_style_str)
        except Exception as e:
            print(f"Error loading {file_path}: {e}. Skipping file.")
            continue

        gas_atoms = u.select_atoms(f'type {gas_atom_type}')
        
        if len(gas_atoms) == 0:
            print(f"WARNING: No gas atoms found with type {gas_atom_type} in {file_path}. Skipping.")
            continue
            
        if len(Y_LABELS) != len(u.trajectory):
            print(f"WARNING: Label count ({len(Y_LABELS)}) does not match frame count ({len(u.trajectory)}) in {file_path}. Using only the minimum number of frames.")
            
        for i, ts in tqdm(enumerate(u.trajectory), total=min(len(u.trajectory), len(Y_LABELS)), 
                           desc=f"Voxelizing Frames in {os.path.basename(file_path)}"):
            
            if i >= len(Y_LABELS):
                break
                
            current_coords = ts.positions
            box_dims = ts.dimensions[:3]
            
            gas_indices = gas_atoms.indices
            center = find_bubble_center(current_coords, gas_indices, box_dims)
            
            voxel_grid = voxelize_bubble(current_coords, center, box_dims)
            X_ALL.append(voxel_grid)
            Y_ALL.append(Y_LABELS[i])
            
    X_ALL = np.array(X_ALL, dtype=np.float32)
    Y_ALL = np.array(Y_ALL, dtype=np.int64) 
    
    print("\n--- Data Loading and Voxelization Complete ---")
    print(f"Total Samples (Frames Processed): {len(X_ALL)}")
    
    return X_ALL, Y_ALL

def load_mock_trajectories(num_frames=1000):
    print("\n--- Running Fallback Mock Data Generation ---")
    X = []
    y = [] 
    
    DEFAULT_BOX_DIMS = np.array([275.0, 275.0, 275.0]) 

    for i in tqdm(range(num_frames), desc="Processing Mock Frames"):
        num_atoms = 1000
        gas_count = 50
        coords = np.random.rand(num_atoms, 3) * DEFAULT_BOX_DIMS
        gas_indices = np.arange(gas_count)
        
        mock_center = DEFAULT_BOX_DIMS / 2.0
        coords[gas_indices] = mock_center + np.random.randn(gas_count, 3) * 2.0 
        
        current_class = random.randint(0, NUM_CLASSES - 1)
        
        center = find_bubble_center(coords, gas_indices, DEFAULT_BOX_DIMS)
        voxel_grid = voxelize_bubble(coords, center, DEFAULT_BOX_DIMS)
        
        X.append(voxel_grid)
        y.append(current_class)
        
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64) 
    
    return X, y

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    print("\n--- Starting Model Training ---")
    model.train()
    for epoch in range(num_epochs):
        running_loss, correct_predictions, total_samples = 0.0, 0, 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        model.train()
        
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_predictions / total_samples
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total

        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss, correct, total = 0.0, 0, 0
    predictions_list, labels_list = [], []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predictions_list.extend(predicted.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    avg_loss = test_loss / len(test_loader)
    
    print("\n--- Evaluating Model on Test Data ---")
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    return np.array(predictions_list), np.array(labels_list)

def run_predictor():
    
    # --- IMPORTANT: CONFIGURATION SECTION ---
    # 1. Provide a list of all your dump file paths.
    DUMP_FILES = ["trajectory_A.dump", "trajectory_B.dump", "trajectory_C.dump"] 
    GAS_ATOM_TYPE = 1 
    
    # 2. Manually create the labels (NumPy arrays) for each corresponding file.
    #    0=Stable, 1=Coalescing, 2=Dissolving, 3=Peanut
    LABELS_A = np.array([0] * 50 + [1] * 50) 
    LABELS_B = np.array([2] * 20 + [0] * 80) 
    LABELS_C = np.array([3] * 100)           
    
    Y_LABELS_LIST = [LABELS_A, LABELS_B, LABELS_C]
    # ----------------------------------------
    
    model = ConvNet3D(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- Data Loading and Splitting ---
    X, y = load_and_process_trajectories(DUMP_FILES, Y_LABELS_LIST, GAS_ATOM_TYPE)

    if len(X) < 10:
        print("\nAborting model run because not enough data was loaded (min 10 frames).")
        return

    full_dataset = BubbleDataset(X, y)

    # Standardized data split
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_val_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=generator)

    val_size = int(0.1 * len(train_val_dataset))
    train_size = len(train_val_dataset) - val_size
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Model Training or Loading ---
    model = ConvNet3D(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model_loaded = load_model(model, MODEL_PATH, DEVICE)

    if not model_loaded:
        print(f"Training Samples: {len(train_dataset)}")
        print(f"Validation Samples: {len(val_dataset)}")
        print(f"Test Samples: {len(test_dataset)}")
        print(f"Using device: {DEVICE}")
        print(model) 
        
        train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS)
        save_model(model, MODEL_PATH)

    # --- Evaluation ---
    predictions, true_labels = evaluate_model(model, test_loader, criterion)
    
    print("\n--- Sample Prediction Demonstration ---")
    
    sample_indices = random.sample(range(len(true_labels)), min(5, len(true_labels)))
    
    for i in sample_indices:
        print(f"Test Sample: {i}")
        print(f"  True Fate: {CLASS_NAMES[true_labels[i]]}")
        print(f"  Predicted Fate: {CLASS_NAMES[predictions[i]]}")
        
if __name__ == '__main__':
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    run_predictor()
