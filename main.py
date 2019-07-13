import os

from Dataset import CUB_bird
from config import Config
from FUNIT import FSUGAN

config = Config()
os.environ['CUDA_VISIBLE_DEVICES']= str(config.gpu_id)

if __name__ == "__main__":

    d_ob = CUB_bird(config)
    dwgan = FSUGAN(d_ob, config)
    dwgan.build_model()

    if config.is_training:
        dwgan.train()






