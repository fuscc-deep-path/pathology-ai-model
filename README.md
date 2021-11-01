# Pathology AI Model

[DeepPath Repo](https://github.com/fuscc-deep-path)

## Installation

```bash
# Python >= 3.7
conda create -n deeppath python=3.9
conda activate deeppath
git clone https://github.com/fuscc-deep-path/pathology-ai-model.git

cd pathology-ai-model
python setup.py sdist
pip install dist/pathology-ai-model-*.tar.gz

# Install PyTorch with CUDA
pip install -f https://download.pytorch.org/whl/cu111/torch_stable.html torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0

# Or Install PyTorch with CPU
pip install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
```

## Usage

```bash
Usage: pathology_ai_model [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  detector       To detect tumor type from image patch.
  normalization  To normalize the image patches.
  sampling       To sample several image patches.
  slide2patch    To convert slide to several patches.
  heatmap        To make a heatmap for the selected image patches.
  prediction     To predict with the specified model.
```
