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

def get_task_ID(patch_num_base=250, scores_cutoff = None):
    if scores_cutoff is None:
        task_ID = 'alltumor' + '_patch' + str(patch_num_base)
    else:
        task_ID = 'cutoff' + str(scores_cutoff) + '_patch' + str(patch_num_base)
    print(task_ID)
    return task_ID


def start_sampling(datapath, featsfile, resultpath, seed=2020, subtype='BLIS', scores_cutoff=None, patch_num_base=250):
    """
    subtype: 'BLIS', 'IM', 'LAR',  'MES'
    """
    np.random.seed(seed)
    task = 'mRNA_subtype_' + subtype
    task_ID = get_task_ID(patch_num_base, scores_cutoff)

    sample_id = os.path.basename(datapath.strip('/'))
    writepath = os.path.join(resultpath, task)
    if not os.path.exists(writepath):
        os.makedirs(writepath)

    with open(os.path.join(writepath, task_ID + '_test.txt'), 'w') as f:
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
