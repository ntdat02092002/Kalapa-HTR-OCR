from vietocr.tool.predictor import Predictor
from vietocr.tool.translate import build_model
from vietocr.tool.utils import download_weights
from vietocr.tool.config import Cfg
import os
import torch

def custom_init(self, config):

    device = config['device']

    model, vocab = build_model(config)
    weights = '/tmp/weights.pth'

    if config['weights'].startswith('http'):
        weights = download_weights(config['weights'])
    else:
        weights = config['weights']

    model.load_state_dict(torch.load(weights, map_location=torch.device(device))['state_dict'])

    self.config = config
    self.model = model
    self.vocab = vocab
    self.device = device

config = Cfg.load_config_from_file("config.yml")
config['weights'] = 'vgg_seq2seq.pth'
config['device'] = 'cpu'

Predictor.__init__ = custom_init
detector = Predictor(config)

torch.save(detector.model, "my-model.pth")
