import numpy as np
import torch
import torch.nn as nn
import os
from tqdm import tqdm

try:
    import MDAnalysis as mda
except ImportError:
    print("MDAnalysis not installed.")
    mda = None 

# --- CONFIGURATION (Must match training exactly) ---
VOXEL_GRID_SIZE = 64                  
VOXEL_CUBE_HALF_SIDE = 80.0           
TIME_WINDOW = 5 
NUM_CLASSES = 3
CLASS_NAMES = ["Stable/Spherical", "Coalescence Imminent", "Dissolution/Collapse"]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "bubble_fate_predictor_model.pth" 
NEW_DUMP_FILE = "dump_test_unknown.lammpstrj" 
GAS_ATOM_TYPE = 3

# --- Model Architecture ---
class ConvNet3D(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet3D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(TIME_WINDOW, 32, kernel_size=3, padding=1),
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

# --- Helper Functions ---
def find_bubble_center(frame_coords, gas_indices, box_dims):
    if gas_indices.size == 0: return box_dims / 2.0
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
    
    flattened_indices = (voxel_indices[:, 0] * VOXEL_GRID_SIZE**2) + \
                        (voxel_indices[:, 1] * VOXEL_GRID_SIZE) + \
                        voxel_indices[:, 2]
    
    counts = np.bincount(flattened_indices, minlength=VOXEL_GRID_SIZE**3)
    voxel_grid = counts.reshape((VOXEL_GRID_SIZE, VOXEL_GRID_SIZE, VOXEL_GRID_SIZE)).astype(np.float32)

    max_count = np.max(voxel_grid)
    if max_count > 0: voxel_grid /= max_count
    return voxel_grid

def load_inference_data(file_path, gas_atom_type):
    if mda is None: return None
    print(f"\n--- Processing Inference File: {file_path} ---")
    atom_style_str = 'id type x y z vx vy vz c_dstress[1] c_dstress[2] c_dstress[3]'
    
    try:
        u = mda.Universe(file_path, format='LAMMPS', atom_style=atom_style_str)
    except Exception as e:
        print(f"Error: {e}")
        return None

    gas_atoms = u.select_atoms(f'type {gas_atom_type}')
    file_voxel_grids = []
    
    for ts in tqdm(u.trajectory, desc="Voxelizing"):
        center = find_bubble_center(ts.positions, gas_atoms.indices, ts.dimensions[:3])
        voxel_grid = voxelize_bubble(ts.positions, center, ts.dimensions[:3])
        file_voxel_grids.append(voxel_grid)
        
    X_data = []
    if len(file_voxel_grids) >= TIME_WINDOW:
        num_windows = len(file_voxel_grids) - TIME_WINDOW + 1
        for j in range(num_windows):
            window_stack = np.stack(file_voxel_grids[j : j + TIME_WINDOW], axis=0)
            X_data.append(window_stack)
    
    return np.array(X_data, dtype=np.float32)

def run_inference():
    if not os.path.exists(MODEL_PATH):
        print("Error: Model file not found! Train the model first.")
        return

    print(f"Loading model from {MODEL_PATH}...")
    model = ConvNet3D(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    X = load_inference_data(NEW_DUMP_FILE, GAS_ATOM_TYPE)
    if X is None or len(X) == 0:
        print("No data processed.")
        return

    inputs = torch.from_numpy(X).float().to(DEVICE)
    print(f"\nRunning inference on {len(inputs)} windows...")
    
    predictions, probs = [], []
    with torch.no_grad():
        for i in range(0, len(inputs), 8): # Batch size 8
            batch = inputs[i : i + 8]
            outputs = model(batch)
            batch_probs = torch.nn.functional.softmax(outputs, dim=1)
            _, batch_preds = torch.max(outputs, 1)
            predictions.extend(batch_preds.cpu().numpy())
            probs.extend(batch_probs.cpu().numpy())

    print(f"\n--- Predictions for {NEW_DUMP_FILE} ---")
    for i, pred in enumerate(predictions):
        frame_idx = i + TIME_WINDOW - 1
        confidence = probs[i][pred] * 100
        print(f"Window ending Frame {frame_idx}: {CLASS_NAMES[pred]} ({confidence:.1f}%)")

if __name__ == '__main__':
    run_inference()
