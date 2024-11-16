# Deer-Detection

## Raspberry Pi 4 Setup
1. Setup the Raspberry Pi with the 64-bit OS.
2. Connect the Raspberry Pi to the internet.
3. Go to System Preferences -> Raspberry Pi Configuration -> Interfaces and enable all of the options except for VNC.

### Miniconda Setup
1. Within terminal, run sudo wget https://github.com/conda-forge/miniforge/releases/download/24.9.2-0/Mambaforge-24.9.2-0-Linux-aarch64.sh.
2. Then, run:
```
bash Mambaforge-24.9.2-0-Linux-aarch64.sh
```
3. Within the instructions while installing, say yes to automatically initializing conda.
4. Close the current terminal and open up another one, and conda should be installed.
5. Finally, create the conda environment using the command below:
```
conda create -n embedded_deer python=3.7
```

### Conda Environment Setup
1. Git clone the grove.py repository by running:
```
git clone https://github.com/Seeed-Studio/grove.py
```
2. Go into the grove.py repo and pip install the packages within it by running:
```
cd grove.py
pip3 install .
```
3. Finally, install the MLX90640 driver with the following command:
```
pip3 install seeed-python-mlx90640
```
