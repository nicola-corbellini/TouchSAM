# TouchSAM
## 🎨 Overview
The FastSAM TouchDesigner Plugin is a .tox component that integrates real-time segmentation into TouchDesigner using the [FastSAM model](https://docs.ultralytics.com/models/fast-sam/).

## Requirements (manual setup)
> [!WARNING]
> This is for advanced users only as it requires installing CUDA and the Python virtual environment from scratch.

<details>
  <summary>Manual setup</summary>

  1. Install Python 3.11.x (the higher x, the better)
  2. Install [CUDA Toolkit](https://developer.nvidia.com/cuda-11-8-0-download-archive) 11.8
  3. Install the required packages
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 ultralytics
  ```
  4. Proceed with the `Installation` steps
</details>

## 🚀 Installation
1. Download the Python environment with pre-installed packages from [here](https://drive.google.com/file/d/1cJnc45tdVYzpREvWfON3MRuD5XWBkSCO/view?usp=sharing)
2. Unzip the folder
> [!NOTE]
> You can skip points 1. and 2. if you followed the `manual setup` steps.
3. **Download** the `.tox` file from the [GitHub Releases]().
4. **Drag & Drop** the `.tox` file into your TouchDesigner project.

## Inputs & Outputs

### Inputs:
- Image or video (TOP): input image or video to segment (e.g. from a `Movie File In TOP`)
### Outputs:
- Segmented image (TOP): processed image with segmentation masks applied.
- Mask image (TOP): black and white specified mask.
- Result table (DAT): comprehensive table with resulting segmentation masks' properties. E.g.:

[//]: # (put a table)

- Logs table (DAT): table with Python logs storing errors and messages from the Python console.

## Usage

### Inference

### Parameters

## Examples

## Roadmap
[] Implement prompting modalities (i.e., points, boxes and text)
[] Add support for other models (e.g., MobileSAM, SAM2)
[] Add support for [TopArray](https://github.com/IntentDev/TopArray) interface to improve performance
