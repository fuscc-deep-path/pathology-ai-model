#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""
    pathology_ai_model.cli
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    AI models for pathology images.

    :copyright: © 2019 by the Choppy Team.
    :license: AGPLv3+, see LICENSE for more details.
"""

"""Console script for pathology_ai_model."""

import sys
from os.path import abspath, dirname

root_dir = dirname(dirname(abspath(__file__)))
sys.path.insert(0, root_dir)

import click
from pathology_ai_model.tumor_detector import start_model as start_detector_model
from pathology_ai_model.sampling import start_sampling
from pathology_ai_model.prediction import start_model as start_prediction_model

@click.group()
def cli():
    pass

@cli.command()
@click.option('--datapath', '-d', required=True, help="The directory which saved normalized image patches.", type=click.Path(exists=True, dir_okay=True))
@click.option('--feats-path', '-f', required=True, help="The directory which to save feats files.", type=click.Path(exists=True, dir_okay=True))
def detector(datapath, feats_path):
    """Detect tumor type from image patch."""
    start_detector_model(datapath, feats_path)

@cli.command()
@click.option('--datapath', '-d', required=True, help="The directory which saved normalized image patches.", type=click.Path(exists=True, dir_okay=True))
@click.option('--feats-file', '-f', required=True, help="The file which saved feats (npz file).", type=click.Path(exists=True, file_okay=True))
@click.option('--result-path', '-r', required=True, help="The directory which to save result files.", type=click.Path(exists=True, dir_okay=True))
@click.option('--seed', '-S', required=False, help="Random seed (default: 2020).", default=2020)
@click.option('--subtype', '-s', required=False, help="Subtype of the tumor (default: BLIS).", default='BLIS', type=click.Choice(['BLIS', 'IM', 'LAR',  'MES']))
@click.option('--scores-cutoff', '-c', required=False, help="Cutoff value for score (default: None).", default=None, type=int)
@click.option('--patch-num-base', '-n', required=False, help="How many patches (default: 250)?", default=250, type=int)
def sampling(datapath, feats_file, result_path, seed, subtype, scores_cutoff, patch_num_base):
    """Sampling several image patches."""
    start_sampling(datapath, featsfile=feats_file, resultpath=result_path, seed=seed, subtype=subtype, scores_cutoff=scores_cutoff, patch_num_base=patch_num_base)


@cli.command()
@click.option('--datapath', '-d', required=True, help="The directory which saved normalized image patches.", type=click.Path(exists=True, dir_okay=True))
@click.option('--sampling-file', '-f', required=True, help="The file which saved sampling images.", type=click.Path(exists=True, file_okay=True))
@click.option('--result-path', '-r', required=True, help="The directory which to save result files.", type=click.Path(exists=True, dir_okay=True))
@click.option('--seed', '-S', required=False, help="Random seed (default: 2020).", default=2020)
@click.option('--gpu', '-g', required=False, help="Which gpu(s) (default: '0')?", default='0')
@click.option('--net', required=False, help="Which net (default: resnet18)?", default='resnet18', type=click.Choice(['resnet18', 'alexnet', 'resnet34',  'inception_v3']))
@click.option('--num-classes', '-n', required=False, help="How many classes (default: 2)?", default=2, type=int)
@click.option('--num-workers', '-N', required=False, help="How many workers (default: 4)?", default=4, type=int)
@click.option('--batch-size', '-b', required=False, help="Batch size (default: 256)?", default=256, type=int)
def prediction(datapath, sampling_file, result_path, seed, gpu, net, num_classes, num_workers, batch_size):
    """Prediction."""
    start_prediction_model(datapath, sampling_file=sampling_file, result_path=result_path, seed=seed, gpu=gpu, net=net, num_classes=num_classes, num_workers=num_workers, batch_size=batch_size)


main = click.CommandCollection(sources=[cli])

if __name__ == '__main__':
    main()
