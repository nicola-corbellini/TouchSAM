# TouchSAM
## 🎨 Overview
The FastSAM TouchDesigner Plugin is a .tox component that integrates real-time segmentation into TouchDesigner using the [FastSAM model](https://docs.ultralytics.com/models/fast-sam/).

## Requirements (manual installation)

<details>
  <summary>Portable (<ins>recommended</ins>)</summary>

1. Download the Python environment with pre-installed packages from [here](https://drive.google.com/file/d/1cJnc45tdVYzpREvWfON3MRuD5XWBkSCO/view?usp=sharing)
2. Unzip the folder
</details>

<details>
  <summary>Manual (advanced)</summary>

  1. Install Python 3.11.x (the higher x, the better)
  2. Install [CUDA Toolkit](https://developer.nvidia.com/cuda-11-8-0-download-archive) 11.8
  4. Install the required packages
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 ultralytics
  ```
</details>

## 🚀 Installation
1. **Download** the `.tox` file from the [GitHub Releases]().
2. **Drag & Drop** the `.tox` file into your TouchDesigner project.
3. Installed the 
