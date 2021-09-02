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
from pathology_ai_model.slide2patch import run_slide2patch as start_slide2patch
from pathology_ai_model.stain_norm import start_stain_norm
from pathology_ai_model.heatmap import make_heatmap as start_heatmap

@click.group()
def cli():
    pass

@cli.command()
@click.option('--datapath', '-d', required=True,
              help="The directory which saved normalized image patches.",
              type=click.Path(exists=True, dir_okay=True))
@click.option('--feats-path', '-f', required=True,
              help="The file which to save features.")
def detector(datapath, feats_path):
    """To detect tumor type from image patch."""
    start_detector_model(datapath, feats_path)

@cli.command()
@click.option('--datapath', '-d', required=True,
              help="The directory which saved normalized image patches.",
              type=click.Path(exists=True, dir_okay=True))
@click.option('--feats-file', '-f', required=True,
              help="The file which saved feats (npz file).",
              type=click.Path(exists=True, file_okay=True))
@click.option('--sampling-file', '-s', required=True,
              help="The file which to save results.")
@click.option('--seed', '-S', required=False, default=2020,
              help="Random seed (default: 2020).")
@click.option('--scores-cutoff', '-c', required=False,
              help="Cutoff value for score (default: None).",
              default=None, type=int)
@click.option('--patch-num-base', '-n', required=False,
              help="How many patches (default: 250)?",
              default=250, type=int)
def sampling(datapath, feats_file, sampling_file, seed, scores_cutoff, patch_num_base):
    """To sample several image patches."""
    start_sampling(datapath, feats_file, sampling_file, seed=seed, scores_cutoff=scores_cutoff, patch_num_base=patch_num_base)


@cli.command()
@click.option('--datapath', '-d', required=True,
              type=click.Path(exists=True, dir_okay=True),
              help="The directory which saved normalized image patches.")
@click.option('--sampling-file', '-f', required=True,
              type=click.Path(exists=True, file_okay=True),
              help="The file which saved sampling images.")
@click.option('--root-dir', '-r', required=True,
              type=click.Path(exists=True, dir_okay=True),
              help="The root directory which to save result files.")
@click.option('--model-type', '-m', required=False, 
              help="The model type for prediction (default: PIK3CA_Mutation).",
              default='PIK3CA_Mutation', type=click.Choice(['PIK3CA_Mutation', 'BLIS', 'IM', 'LAR',  'MES']))
@click.option('--seed', '-S', required=False,
              help="Random seed (default: 2020).", default=2020)
@click.option('--gpu', '-g', required=False,
              help="Which gpu(s) (default: '0')?", default='0')
@click.option('--net', required=False, type=click.Choice(['resnet18', 'alexnet', 'resnet34',  'inception_v3']),
              help="Which net (default: resnet18)?", default='resnet18')
@click.option('--num-classes', '-n', required=False,
              help="How many classes (default: 2)?", default=2, type=int)
@click.option('--num-workers', '-N', required=False,
              help="How many workers (default: 4)?", default=4, type=int)
@click.option('--batch-size', '-b', required=False,
              help="Batch size (default: 256)?", default=256, type=int)
def prediction(datapath, sampling_file, root_dir, model_type, seed, gpu, net, num_classes, num_workers, batch_size):
    """To predict with the specified model."""
    start_prediction_model(datapath, sampling_file=sampling_file, root_dir=root_dir, model_type=model_type, seed=seed, gpu=gpu, net=net, num_classes=num_classes, num_workers=num_workers, batch_size=batch_size)


@cli.command()
@click.option('--xmlpath', '-x', required=True, type=click.Path(exists=True, file_okay=True), help="The xml file for ROI.")
@click.option('--output', '-o', required=True, help="The output directory for saving image patches.")
def slide2patch(xmlpath, output):
    """To convert slide to several patches."""
    start_slide2patch(xmlpath, output)


@cli.command()
@click.option('--datapath', '-d', required=True, type=click.Path(exists=True, file_okay=True), help="The directory which contains non-normalized patches.")
@click.option('--output', '-o', required=True, help="The output directory for saving normalized image patches.")
def normalization(datapath, output):
    """To normalize the image patches."""
    start_stain_norm(datapath, output)


@cli.command()
@click.option('--datapath', '-d', required=True,
              type=click.Path(exists=True, dir_okay=True),
              help="The directory which saved normalized image patches.")
@click.option('--feats-file', '-f', required=True,
              help="The file which saved feats (npz file).",
              type=click.Path(exists=True, file_okay=True))
@click.option('--sampling-file', '-s', required=True,
              type=click.Path(exists=True, file_okay=True),
              help="The file which saved sampling images.")
@click.option('--root-dir', '-r', required=True,
              type=click.Path(exists=True, dir_okay=True),
              help="The root directory which to save result files.")
@click.option('--model-type', '-m', required=False, 
              help="The model type for prediction (default: PIK3CA_Mutation).",
              default='PIK3CA_Mutation', type=click.Choice(['PIK3CA_Mutation', 'BLIS', 'IM', 'LAR',  'MES']))
@click.option('--gpu', '-g', required=False,
              help="Which gpu(s) (default: '0')?", default='0')
@click.option('--num-classes', '-n', required=False,
              help="How many classes (default: 2)?", default=2, type=int)
def heatmap(datapath, feats_file, sampling_file, root_dir, model_type, gpu, num_classes):
    """To make a heatmap for the selected image patches."""
    start_heatmap(datapath, featsfile=feats_file, sampling_file=sampling_file, root_dir=root_dir, model_type=model_type, gpu=gpu, num_classes=num_classes)



main = click.CommandCollection(sources=[cli])

if __name__ == '__main__':
    main()
