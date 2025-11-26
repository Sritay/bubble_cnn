# Deep Learning for Bubble Fate Prediction in MD

This project implements a **3D Spatio-Temporal Convolutional Neural Network (CNN)** to classify the stability fate of nanobubbles from Molecular Dynamics (MD) simulations. It processes raw LAMMPS trajectories, converts atomic coordinates into volumetric density grids, and predicts whether a bubble is **Stable**, **Collapsing**, or **Coalescing**.

## Key Features

* **End-to-End Pipeline:** Reads raw LAMMPS dump files using `MDAnalysis` and outputs classification predictions.
* **3D Voxelization:** Converts continuous particle positions into discrete 3D density tensors ($64 \times 64 \times 64$ grid).
* **Spatio-Temporal Learning:** Stacks $T=5$ consecutive frames to allow the model to learn from motion and velocity, not just static shape.
* **HPC Ready:** Includes `Singularity` definition files and `Slurm` scripts for deployment on clusters like ARCHER2.

## Usage

### Prerequisites

* Python 3.10+
* PyTorch (CPU or CUDA)
* MDAnalysis, NumPy, Tqdm

### Running Training

1.  Configure the `DUMP_FILES` and `Y_LABELS_LIST` in `bubble_fate_predictor.py`.
2.  Run:
    ```bash
    python bubble_fate_predictor.py
    ```
    *This will train the model, save the weights to `bubble_fate_predictor_model.pth`, and output evaluation metrics.*

### Running Inference

To predict the fate of a new, unlabeled trajectory:

1.  Ensure the trained model (`.pth`) is present.
2.  Update `NEW_DUMP_FILE` in `predict_fate.py`.
3.  Run:
    ```bash
    python predict_fate.py
    ```

## Model Limitations & Caveats

**1. Dataset Size & Accuracy**
The current model achieves **100% accuracy** on the test set. This is a "Proof of Concept" result derived from a small dataset (~300 windows).
* **Data Leakage:** The training and test sets were split randomly from the same continuous trajectories. Due to the sliding window approach ($T=5$), there is significant overlap between frames in the training and test sets, helping the model "memorize" sequences.
* **Generalization:** To prove true generalization, the model should be tested on entirely new trajectories that were not seen during training.

**2. Field of View**
The voxel grid is fixed to a **160 Å ($16 \text{ nm}$)** box width. Bubbles larger than this diameter will be cropped, potentially losing surface information.

**3. Density-Based Voxelization**
The current voxelizer counts *all* atoms (liquid and gas) to create a density map. It does not separate them into different channels. The model relies on the density contrast between the gas core and the surrounding liquid to identify shapes.
