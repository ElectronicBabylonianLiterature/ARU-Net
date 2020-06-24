from __future__ import print_function, division
import click
from pix_lab.util.predict import Predict_pb
from pix_lab.util.util import read_image_list


@click.command()
@click.option('--path_list_imgs', default="./predict/pred")
@click.option('--path_net_pb', default="./model_ema/model100_ema.pb")
def run(path_list_imgs, path_net_pb):
    list_inf = read_image_list(path_list_imgs)
    inference = Predict_pb(path_net_pb)
    outputs = inference.predict(list_inf)
    print("Done")



if __name__ == '__main__':
    run()