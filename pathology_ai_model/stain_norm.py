import glob
import os
import pkgutil
# import numpy as np

from pathology_ai_model.common.stain_step import run_stainsep
from pathology_ai_model.common.color_norm_pytorch import ColorNorm


def write_reference(data, output):
    with open(output, 'wb') as f:
        f.write(data)


def start_stain_norm(datapath, output, stain_norm = True, nstains = 2, lamb = 0.01):
    """
    Arguments:
      stain_norm: If True, color normalization of images in a folder with one target image, else, stain separation
      nstains: Number of stains
      lamb: default value sparsity regularization parameter  # lamb=0 equivalent to NMF
    """
    data = pkgutil.get_data(__package__, 'reference/FUSCCTNBC089_81_401_1.png')
    reference_filepath = os.path.join(os.path.dirname(output), 'FUSCCTNBC089_81_401_1.png')
    write_reference(data, reference_filepath)

    assert os.path.exists(reference_filepath), "Target file does not exist"

    img_format = "*.png"
    filenames = sorted(glob.glob(os.path.join(datapath, img_format)))
    assert len(filenames) != 0, "No source files found"

    if not stain_norm:
        pass
    else:
        if not os.path.exists(output):
            os.makedirs(output)

        filenames = [reference_filepath] + filenames
        norm = ColorNorm(nstains, lamb, img_level=0)
        norm.run_batch_colornorm(filenames, output)
