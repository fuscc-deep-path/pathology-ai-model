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
import click

from pathology_ai_model.tumor_detector import start_model as start_detector_model

@click.group()
def cli():
    pass

@cli.command()
@click.option('--datapath', '-d', required=True, help="The directory which saved normalized image patches.", type=click.Path(exists=True))
@click.option('--feats-path', '-f', required=True, help="The directory which to save feats files.", type=click.Path(exists=True))
def detector(datapath, feats_path):
    """Detect tumor type from image patch."""
    start_detector_model(datapath, feats_path)

@cli.command()
def sampling():
  """Sampling several image patches."""

main = click.CommandCollection(sources=[cli])

if __name__ == '__main__':
    main()
