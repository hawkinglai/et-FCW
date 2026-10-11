# Tested environment

| Component | Version |
|---|---|
| Python | 3.12.2 |
| PyTorch / CUDA | 2.7.1+cu118 / 11.8 |
| NVIDIA driver / GPU | 560.35.03 / RTX A5000 (architecture 8.6) |
| NumPy / h5py / tqdm | 1.26.4 / 3.11.0 / 4.66.5 |
| scikit-learn / pandas | 1.7.2 / 2.3.2 |
| pointnet2_ops | 3.0.0 |
| PyTorch3D | 0.7.8, commit 366eff21d9d1853a87961129625adb42032af665 |
| iopath / portalocker | 0.1.10 / 4.4.0 |

Install commands (CUDA 11.8 toolkit required for native extensions):

```bash
conda create -n pytorch -c conda-forge python=3.12.2
conda activate pytorch
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install numpy==1.26.4 h5py==3.11.0 tqdm==4.66.5 scikit-learn==1.7.2 pandas==2.3.2 iopath==0.1.10 portalocker==4.4.0
CUDA_HOME=/usr/local/cuda-11.8 TORCH_CUDA_ARCH_LIST=8.6   python -m pip install --no-build-isolation 'git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#subdirectory=pointnet2_ops_lib'
CUDA_HOME=/usr/local/cuda-11.8 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.6   python -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/pytorch3d.git@366eff21d9d1853a87961129625adb42032af665'
```

The tested pointnet2_ops binary was pre-existing; its build revision was not recorded. Its command above identifies the upstream build source, not an exact binary lock. PyTorch3D is installed directly in the environment; RISP acceptance passed with PYTHONPATH unset. Rebuild native extensions for another CUDA/GPU architecture.

Configure dataset variables from [DATA.md](DATA.md), activate this environment, and run each experiment's `run.sh`. Keep recorded batch sizes, workers, seeds and FPS caches.
