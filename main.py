__author__ = 'Weili Nie'

import os
from MNIST import GAN_MNIST
from CelebA import GAN_CelebA
from GMM import GAN_GMM
from RenderData import GAN_RenderData
from config import get_config
from utils import prepare_dirs


config, unparsed = get_config()
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
prepare_dirs(config, config.dataset_img)

if config.dataset_img == 'CelebA':
    gan_model = GAN_CelebA.GAN_model(config)
elif config.dataset_img == 'MNIST':
    gan_model = GAN_MNIST.GAN_model(config)
elif config.dataset_img == 'GMM':
    gan_model = GAN_GMM.GAN_model(config)
elif config.dataset_img in ['RenderBall', 'RenderBallTri']:
    gan_model = GAN_RenderData.GAN_model(config)
else:
    raise Exception("[!] Unrecognized dataset_name!")

if config.is_train:
    gan_model.train()
else:
    if not config.load_path:
        raise Exception("[!] You should specify `load_path` to load a pretrained model")
    gan_model.test()