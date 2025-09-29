# Running Mitsui Commodity LightGBM Pipeline on a Fresh Machine

This guide walks through setting up the repository when you only have the raw CSV files (no pre-generated artifacts). It covers environment preparation, feature generation, and running the LightGBM baselines on CPU. GPU-specific notes are included at the end if you want to accelerate training.

## 1. Prerequisites
- **OS**: Windows (PowerShell) or Linux (Bash). macOS is possible but CUDA/GPU instructions differ.
- **Python**: 3.10 or newer.
- **Git**: to clone the repository.
- **Build tools** (optional, for GPU LightGBM build):
  - Linux: `build-essential`, `cmake`, `ninja-build`, `libboost-dev`, CUDA toolkit (if using GPU).
  - Windows: Visual Studio Build Tools with CMake & Ninja, CUDA toolkit.

## 2. Clone the Repository and Stage Data
```bash
git clone https://github.com/lanyama19/Mitsui-Commodity-Pred.git
cd Mitsui-Commodity-Pred
```
Place the original CSVs in the repository root (same directory as `README.md`):
```
train.csv
test.csv
train_labels.csv
target_pairs.csv
```
They are already git-ignored, so they won’t be committed accidentally.

## 3. Create a Virtual Environment and Install Dependencies
### 3.1 Create & activate the environment
```bash
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / WSL / macOS
python -m venv .venv
source .venv/bin/activate
```

### 3.2 Install Python dependencies
The project includes `requirements.txt`. Install everything with pip; PyTorch requires a custom index (adjust for your CUDA version—below uses CUDA 12.6 wheels). If you prefer CPU-only, see the alternative command.

```bash
# CUDA 12.6 build (recommended if you have a compatible NVIDIA GPU)
pip install --upgrade pip
pip install torch==2.8.0+cu126 torchvision==0.23.0+cu126 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# (Optional) CPU-only PyTorch instead of the CUDA wheels
# pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
```

> **TA-Lib note**: The wheel in `requirements.txt` works on Win/Linux. If it fails on Linux, install the system package first (e.g., `sudo apt install ta-lib`) then rerun pip.

## 4. Generate Cleaned Price Panels and Features
1. **Clean raw prices**:
   ```bash
   python -c "from src.data.cleaning import build_all_cleaned; build_all_cleaned(window=5)"
   ```
   This writes cleaned price tables to `artifacts/clean_*`. Adjust `window` if you want a different extrapolation horizon.

2. **Build the full feature stack** (technical, factors, PCA, genetic programming):
   ```bash
   python -c "from src.features.pipeline import build_and_save_features; build_and_save_features()"
   ```
   Results appear under `artifacts/features/` (e.g., `all_train.pkl`, `pca.pkl`, `gp.pkl`, etc.). This step can take a while; the script prints progress as each block finishes.

3. (Optional) Inspect the generated features for a given lag:
   ```bash
   python scripts/feature_eda.py --lag 1 --panel all_train.pkl --head 10 --summary-csv artifacts/eda/lag1_summary.csv
   ```

## 5. Run LightGBM Baseline Training
The main entry point is `scripts/run_lightgbm_baseline.py`. Example CPU run for lag 1:
```bash
python scripts/run_lightgbm_baseline.py --lag 1 --run-name cpu_baseline_lag1
```
Key flags:
- `--lag`: selects the lag bucket (1–4).
- `--train-end`: shifts the train/validation split (default 1800).
- `--device`: `cpu` (default) or `gpu` (requires GPU-enabled LightGBM).
- `--run-name`: output directory under `artifacts/lightgbm/`.

Outputs include per-target LightGBM models, predictions, and a `summary.json` with IC/IR metrics.

To train additional lags, rerun with different `--lag` values (1 to 4). You can loop these commands or run them sequentially.

## 6. LightGBM Hyperparameter Grid Search (CPU)
Use `scripts/grid_search_lightgbm.py` to sample hyperparameters and evaluate them. Example CPU sweeps:

```bash
python scripts/grid_search_lightgbm.py --lag 3 --device cpu --n-samples 400 --n-jobs 4 --train-end 1800 --max-train-gap 0.2 --output artifacts/grid_lag3_cpu.csv
python scripts/grid_search_lightgbm.py --lag 4 --device cpu --n-samples 400 --n-jobs 4 --train-end 1800 --max-train-gap 0.2 --output artifacts/grid_lag4_cpu.csv
```
- `--n-samples`: number of random combos from the default grid.
- `--n-jobs`: parallel workers (set according to your CPU cores; on GPU mode this is forced to 1).
- `--max-train-gap`: filters out overfitted configs where train IC far exceeds validation IC.

Outputs: CSV with all evaluated runs and terminal printout of top configurations.

## 7. Optional: Enable GPU-Accelerated LightGBM
If you want LightGBM GPU training, install a GPU-enabled build. On Linux/WSL:

```bash
sudo apt install -y build-essential cmake ninja-build libboost-dev libboost-system-dev libboost-filesystem-dev
# Ensure CUDA toolkit and nvcc are available

cd LightGBM
rm -rf build
cmake -B build -S . -DUSE_CUDA=ON -DCMAKE_C_COMPILER=/usr/bin/gcc-12 -DCMAKE_CXX_COMPILER=/usr/bin/g++-12
cmake --build build -j$(nproc)
cp LICENSE python-package/
CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 \
pip install --no-build-isolation --config-settings=cmake.source-dir=.. \
    --config-settings=cmake.define.USE_CUDA=ON \
    --config-settings=cmake.define.CMAKE_C_COMPILER=/usr/bin/gcc-12 \
    --config-settings=cmake.define.CMAKE_CXX_COMPILER=/usr/bin/g++-12 \
    --config-settings=cmake.define.CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
    ./python-package
```
Replace compiler paths as needed; on Windows, use the MSVC toolchain equivalents. After installation, rerun the baseline or grid search with `--device gpu` and the appropriate `--gpu-platform-id/--gpu-device-id` values.

You can quickly verify the GPU build with:
```bash
python scripts/smoke_lightgbm_gpu.py --lag 1 --n-samples 3 --gpu-platform-id 0 --gpu-device-id 0
```

## 8. Artifact Layout Overview
- `artifacts/clean_*`: cleaned price tables per dataset split.
- `artifacts/features/`: feature panels and PCA/GP artifacts.
- `artifacts/lightgbm/lag_{k}/run_*/`: per-lag LightGBM runs (models, metrics, predictions).
- `artifacts/grid_lag*_*.csv`: captured hyperparameter search results.

Keep these directories if you plan to resume training; otherwise you can delete them to free disk space.

## 9. Reproducibility Tips
- Commit your configuration files or note any overrides to `LightGBMConfig` to reproduce runs.
- Use `--run-name` to label experiments clearly (e.g., `--run-name cpu_lambda_tune`).
- Pin your Python interpreter and dependency versions (the provided `requirements.txt` captures the baseline environment).

Once the above steps are complete, you’re ready to experiment with LightGBM baselines on the new machine.
