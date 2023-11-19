from vietocr.tool.predictor import Predictor
from vietocr.tool.translate import build_model
from vietocr.tool.utils import download_weights
from vietocr.tool.config import Cfg
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import datetime
import argparse
import os
import torch
import pandas as pd

from post_processing import post_process


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

def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--model',
    #     type=str,
    #     choices=['seq2seq', 'transformer'],
    #     default='transformer',
    #     help='Model could be seq2seq or transformer'
    # )
    parser.add_argument('--config', type=str, default='./',
                       help='Path to custom configs file')
    parser.add_argument('--weight', type=str,
                       help='Path to weight')
    parser.add_argument('--directory', help='folder contain images to read')
    parser.add_argument('--output', default='predicts/')
    parser.add_argument('--device', type=str,
                        default='cuda:0', help='Device cuda:0 or cpu')
    return parser.parse_args()

def main():
    args = get_args()

    config = Cfg.load_config_from_file(args.config)
    config['weights'] = args.weight
    config['device'] = args.device

    Predictor.__init__ = custom_init
    detector = Predictor(config)

    directory_path = Path(args.directory)
    output_path = Path(args.output)

    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = output_path / "vietocr" / f"{current_datetime}"
    output_file.mkdir(parents=True, exist_ok=True)

    with open(output_file / "log.txt", 'w+') as log:
        log.write(f'checkpoint path: {args.weight}\n')
        log.write(f'data inference: {directory_path}')
    log.close()

    prediction = pd.DataFrame(columns=['id', 'answer'])
    index = 0

    for root, dirs, files in os.walk(directory_path):
        for fname in tqdm(files, desc="Processing images", unit="image"):
            # Load image and prepare for input
            image_path = os.path.join(root, fname)

            image = Image.open(image_path).convert('RGB')
            pred = detector.predict(image, return_prob=False)
            pred = pred.strip()
            if len(pred) > 80:
                pred = ""

            pred = post_process.correct(pred)

            path_parts = image_path.split(os.sep)
            image_id =  os.path.join(*path_parts[-2:])
            prediction.loc[index] = [image_id, pred]
            index += 1

    prediction.to_csv(output_file / "submission.csv", index=False)
    print("Done!")



if __name__ == "__main__":
    main()

