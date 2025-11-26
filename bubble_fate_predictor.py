import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import os
import random
from tqdm import tqdm

# Attempt to import MDAnalysis for trajectory parsing
try:
    import MDAnalysis as mda
except ImportError:
    print("MDAnalysis not installed. Please run: pip install MDAnalysis")
    mda = None 

# --- CONFIGURATION & PHYSICS PARAMETERS ---
# 1. Voxel Resolution: 64x64x64 grid
#    Resolution = (2 * VOXEL_CUBE_HALF_SIDE) / 64 approx 2.5 Angstroms/voxel
VOXEL_GRID_SIZE = 64                  

# 2. Field of View (FOV): Half-width of the bounding box
#    Set to 80.0 Angstroms (Total box width = 160 A).
#    Chosen to fit large ~14nm (140 A) bubbles with a 10 A liquid buffer.
VOXEL_CUBE_HALF_SIDE = 80.0           

# 3. Batch Size: Reduced to 8 to accommodate large 64^3 tensors in VRAM/RAM.
BATCH_SIZE = 8

# 4. Training Hyperparameters
NUM_EPOCHS = 20
NUM_CLASSES = 3
TIME_WINDOW = 5 # Spatio-Temporal Depth: Stacks 5 consecutive frames
CLASS_NAMES = ["Stable/Spherical", "Coalescence Imminent", "Dissolution/Collapse"]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "bubble_fate_predictor_model.pth" 

# --- Utility Functions ---

def save_model(model, path):
    """Saves the trained model weights."""
    print(f"Saving model to {path}...")
    torch.save(model.state_dict(), path)
    print("Model saved successfully.")

def load_model(model, path, device):
    """Loads model weights if available, enabling restart/inference."""
    if os.path.exists(path):
        print(f"Loading existing model from {path}...")
        model.load_state_dict(torch.load(path, map_location=device))
        print("Model loaded successfully.")
        return True
    else:
        print(f"No existing model found at {path}. Training from scratch.")
        return False
        
def find_bubble_center(frame_coords, gas_indices, box_dims):
    """
    Calculates the Center of Mass (COM) of the gas phase.
    Used to re-center the voxel grid on the bubble, ensuring translational invariance.
    """
    if gas_indices.size == 0:
        return box_dims / 2.0
    gas_coords = frame_coords[gas_indices]
    # Simple mean is sufficient if bubble is not split across PBC
    center = np.mean(gas_coords, axis=0)
    return center

def voxelize_bubble(coords, center, box_dims):
    """
    Converts continuous atomic coordinates into a discrete 3D density grid.
    
    Process:
    1. Centers system on bubble COM.
    2. Applies PBC relative to that center.
    3. Crops to the VOXEL_CUBE_HALF_SIDE (ROI).
    4. Bins atoms into the VOXEL_GRID_SIZE grid.
    """
    # 1. Center & PBC
    coords_centered = coords - center
    coords_pbc = coords_centered - np.round(coords_centered / box_dims) * box_dims
    
    # 2. Crop to Region of Interest (ROI)
    H = VOXEL_CUBE_HALF_SIDE
    mask = (coords_pbc[:, 0] >= -H) & (coords_pbc[:, 0] < H) & \
           (coords_pbc[:, 1] >= -H) & (coords_pbc[:, 1] < H) & \
           (coords_pbc[:, 2] >= -H) & (coords_pbc[:, 2] < H)
           
    cropped_coords = coords_pbc[mask]
    
    if cropped_coords.size == 0:
        return np.zeros((VOXEL_GRID_SIZE, VOXEL_GRID_SIZE, VOXEL_GRID_SIZE), dtype=np.float32)

    # 3. Discretize
    scaling_factor = VOXEL_GRID_SIZE / (2 * H)
    voxel_indices = (cropped_coords + H) * scaling_factor
    voxel_indices = np.clip(voxel_indices, 0, VOXEL_GRID_SIZE - 1).astype(int)
    
    # 4. Density Mapping (Histogram)
    flattened_indices = (voxel_indices[:, 0] * VOXEL_GRID_SIZE**2) + \
                        (voxel_indices[:, 1] * VOXEL_GRID_SIZE) + \
                        voxel_indices[:, 2]
    
    counts = np.bincount(flattened_indices, minlength=VOXEL_GRID_SIZE**3)
    voxel_grid = counts.reshape((VOXEL_GRID_SIZE, VOXEL_GRID_SIZE, VOXEL_GRID_SIZE)).astype(np.float32)

    # Normalize to [0, 1] range for Neural Network stability
    max_count = np.max(voxel_grid)
    if max_count > 0:
        voxel_grid /= max_count
        
    return voxel_grid

class BubbleDataset(Dataset):
    """PyTorch Dataset wrapper for the voxel tensors."""
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ConvNet3D(nn.Module):
    """
    3D Spatio-Temporal Convolutional Neural Network.
    Input: (Batch, Time, Depth, Height, Width)
    """
    def __init__(self, num_classes):
        super(ConvNet3D, self).__init__()
        
        # Block 1: Spatio-Temporal Feature Extraction
        # Input channels = TIME_WINDOW (stacking frames as channels)
        self.conv1 = nn.Sequential(
            nn.Conv3d(TIME_WINDOW, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        
        # Block 2: Higher Level Features (Shape/Topology)
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        
        # Flatten calculation: 64 channels * (64 / 4)^3 spatial size
        self.flatten_size = 64 * (VOXEL_GRID_SIZE // 4)**3
        
        # Classification Head
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5), # Dropout to prevent overfitting on correlated frames
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, self.flatten_size) 
        x = self.fc(x)
        return x

def load_and_process_trajectories(dump_file_paths, y_labels_list, target_frames_list, gas_atom_type=1):
    """
    Core logic to read LAMMPS dumps and generate sliding-window samples.
    Returns X (Features) and y (Labels).
    """
    if mda is None: return [], []
        
    X_ALL = []
    Y_ALL = []
    
    # Atom style includes stress tensors required for some LAMMPS outputs
    atom_style_str = 'id type x y z vx vy vz c_dstress[1] c_dstress[2] c_dstress[3]'
    
    if len(dump_file_paths) != len(y_labels_list):
        print("ERROR: Mismatch between dump files and label lists.")
        return [], []

    if target_frames_list is None:
        target_frames_list = [None] * len(dump_file_paths)

    print(f"\n--- Loading Trajectories (Window Size T={TIME_WINDOW}) ---")

    for file_path, Y_LABELS, TARGET_FRAMES in zip(dump_file_paths, y_labels_list, target_frames_list):
        print(f"Processing: {file_path}")
        
        if TARGET_FRAMES is not None:
            frames_to_process = set(TARGET_FRAMES)
            max_frame = max(TARGET_FRAMES)
        else:
            frames_to_process = None
            max_frame = float('inf')

        try:
            u = mda.Universe(file_path, format='LAMMPS', atom_style=atom_style_str)
        except Exception as e:
            print(f"Error loading {file_path}: {e}. Skipping.")
            continue

        gas_atoms = u.select_atoms(f'type {gas_atom_type}')
        if len(gas_atoms) == 0:
            print(f"WARNING: No gas atoms (Type {gas_atom_type}) in {file_path}. Skipping.")
            continue
            
        # Temporary storage for this file's frames
        file_voxel_grids = []
        file_valid_labels = []
        processed_count = 0
        
        # Frame Loop
        for i, ts in tqdm(enumerate(u.trajectory), total=u.trajectory.n_frames, desc="Voxelizing"):
            # Check if frame is in target list
            if frames_to_process is not None:
                if i not in frames_to_process: continue 
            else:
                if processed_count >= len(Y_LABELS): break

            # Voxelize
            center = find_bubble_center(ts.positions, gas_atoms.indices, ts.dimensions[:3])
            voxel_grid = voxelize_bubble(ts.positions, center, ts.dimensions[:3])
            file_voxel_grids.append(voxel_grid)
            
            # Match Label
            if frames_to_process is not None:
                label_idx = TARGET_FRAMES.index(i)
                file_valid_labels.append(Y_LABELS[label_idx])
            else:
                file_valid_labels.append(Y_LABELS[processed_count])
            
            processed_count += 1
            if frames_to_process is not None and i >= max_frame: break
        
        # Create Sliding Windows (Stack T frames)
        if len(file_voxel_grids) >= TIME_WINDOW:
            num_windows = len(file_voxel_grids) - TIME_WINDOW + 1
            for j in range(num_windows):
                # Stack along axis 0 -> (T, D, D, D)
                window_stack = np.stack(file_voxel_grids[j : j + TIME_WINDOW], axis=0)
                # Label is determined by the last frame in the window
                window_label = file_valid_labels[j + TIME_WINDOW - 1]
                X_ALL.append(window_stack)
                Y_ALL.append(window_label)
        else:
            print(f"WARNING: Not enough frames in {file_path} for window size {TIME_WINDOW}.")

    X_ALL = np.array(X_ALL, dtype=np.float32)
    Y_ALL = np.array(Y_ALL, dtype=np.int64) 
    
    print("\n--- Data Loading Complete ---")
    print(f"Total Windows: {len(X_ALL)}")
    return X_ALL, Y_ALL

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    print("\n--- Starting Training ---")
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Validation Step
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        print(f"Epoch {epoch+1}: Train Acc: {100*correct/total:.1f}% | Val Acc: {100*val_correct/val_total:.1f}%")

def evaluate_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    preds, labs = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            preds.extend(predicted.cpu().numpy())
            labs.extend(labels.cpu().numpy())
    print(f"Test Accuracy: {100*correct/total:.2f}%")
    return np.array(preds), np.array(labs)

def run_predictor():
    
    # --- USER CONFIGURATION ---
    # List your dump files here
    DUMP_FILES = [
        "dump_stable.lammpstrj", 
        "dump_coalesce_1.lammpstrj",
        "dump_coalesce_2.lammpstrj",
        "dump_coalesce_3.lammpstrj",
        "dump_collapse.lammpstrj"
    ]
    GAS_ATOM_TYPE = 3 
    
    # Define frame ranges for each file (matching your dataset)
    FRAMES_SEQ_STABLE = list(range(90))
    FRAMES_SEQ_COALESCE_1 = list(range(25))
    FRAMES_SEQ_COALESCE_2 = list(range(35))
    FRAMES_SEQ_COALESCE_3 = list(range(34))
    FRAMES_SEQ_COLLAPSE = list(range(130))
    
    TARGET_FRAMES_LIST = [
        FRAMES_SEQ_STABLE, 
        FRAMES_SEQ_COALESCE_1, 
        FRAMES_SEQ_COALESCE_2, 
        FRAMES_SEQ_COALESCE_3, 
        FRAMES_SEQ_COLLAPSE
    ]

    # Define labels (Class 0, 1, or 2) for each file
    LABELS_STABLE = np.full(90, 0)
    LABELS_COALESCE_1 = np.full(25, 1)
    LABELS_COALESCE_2 = np.full(35, 1)
    LABELS_COALESCE_3 = np.full(34, 1)
    LABELS_COLLAPSE = np.full(130, 2)
    
    Y_LABELS_LIST = [
        LABELS_STABLE, 
        LABELS_COALESCE_1, 
        LABELS_COALESCE_2, 
        LABELS_COALESCE_3, 
        LABELS_COLLAPSE
    ]
    
    # --- EXECUTION ---
    model = ConvNet3D(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Seed for reproducibility
    split_gen = torch.Generator().manual_seed(42)

    # Check for existing model
    model_loaded = load_model(model, MODEL_PATH, DEVICE)

    if not model_loaded:
        # TRAIN MODE
        X, y = load_and_process_trajectories(DUMP_FILES, Y_LABELS_LIST, TARGET_FRAMES_LIST, GAS_ATOM_TYPE)
        if len(X) < 5: return 
        
        full_dataset = BubbleDataset(X, y)
        
        # 80/10/10 Split
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_val, test_data = random_split(full_dataset, [train_size, test_size], generator=split_gen)
        
        val_size = int(0.1 * len(train_val)) # 10% of total
        train_size = len(train_val) - val_size
        train_data, val_data = random_split(train_val, [train_size, val_size], generator=split_gen)
        
        train_l = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        val_l = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
        test_l = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
        
        train_model(model, train_l, val_l, criterion, optimizer, NUM_EPOCHS)
        save_model(model, MODEL_PATH)
        final_l = test_l
    else:
        # TEST MODE (Load data again to generate test set)
        # Note: In a production script, data should be saved separately to avoid reprocessing.
        X, y = load_and_process_trajectories(DUMP_FILES, Y_LABELS_LIST, TARGET_FRAMES_LIST, GAS_ATOM_TYPE)
        full_dataset = BubbleDataset(X, y)
        
        # Recreate the exact same split to ensure we test on the test set
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        _, test_data = random_split(full_dataset, [train_size, test_size], generator=split_gen)
        final_l = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # Evaluation
    preds, true_labels = evaluate_model(model, final_l)
    
    print("\n--- Sample Predictions ---")
    if len(true_labels) > 0:
        indices = random.sample(range(len(true_labels)), min(3, len(true_labels)))
        for i in indices:
            print(f"True: {CLASS_NAMES[true_labels[i]]} | Pred: {CLASS_NAMES[preds[i]]}")

if __name__ == '__main__':
    torch.manual_seed(42)
    run_predictor()
