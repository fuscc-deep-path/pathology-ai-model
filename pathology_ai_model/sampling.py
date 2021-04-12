# -*- coding: utf-8 -*-
"""
    pathology_ai_model.sampling
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Sampling several image patches.

    :copyright: © 2019 by the Choppy Team.
    :license: AGPLv3+, see LICENSE for more details.
"""

import os
import numpy as np


def start_sampling(datapath, featsfile, samplefile, seed=2020, scores_cutoff=None, patch_num_base=250):
    """
    """
    np.random.seed(seed)
    sample_id = os.path.basename(datapath.strip('/'))
    with open(samplefile, 'w') as f:
        data = np.load(featsfile)
        scores, bins, namelist = data['score'], data['bin'], data['namelist']

        namelist_cut = np.array([os.path.basename(name) for name in namelist])

        if scores_cutoff is None:
            tumor_patchlist = namelist_cut[bins == 0]
        else:
            tumor_patchlist = namelist_cut[scores[:, 0] > scores_cutoff]

        if len(tumor_patchlist) < 20:
            print('Low tumor patch number slide', sample_id, 'tumor patches number:', len(tumor_patchlist))
            return
        else:
            print(sample_id, ' tumor patches number: ', len(tumor_patchlist))

        np.random.shuffle(tumor_patchlist)

        ends = patch_num_base if patch_num_base < len(tumor_patchlist) else len(tumor_patchlist)
        tumor_patchlist_sampled = tumor_patchlist[0:ends]

        for s in tumor_patchlist_sampled:
            name = os.path.join(s + '.png')
            f.write(name + '\n')
