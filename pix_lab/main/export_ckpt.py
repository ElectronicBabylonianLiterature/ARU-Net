from __future__ import print_function, division

import os

import click

from pix_lab.models.aru_net import ARUnet
from pix_lab.util.exporter import Exporter_ckpt


@click.command()
@click.option('--restore_ckt_path', default="../../model_checkpoints/model100")
@click.option('--export_name', default="../../model_ema/model100_ema.pb")
@click.option('--use_ema', default=True)
def run(restore_ckt_path, export_name, use_ema):
    img_channels = 1
    n_class = 2
    model_kwargs = dict(model="aru", final_act="softmax")
    model = ARUnet(img_channels, n_class, model_kwargs=model_kwargs)
    exporter = Exporter_ckpt(model)
    exporter.export(restore_ckt_path, export_name, use_ema_vars=use_ema, output_nodes=['output'])


if __name__ == '__main__':
    run()