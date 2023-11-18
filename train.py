import argparse
from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer
from vietocr.tool.translate import translate, batch_translate_beam_search
from pathlib import Path
import yaml
import datetime
import torch
import time
import numpy as np
import os, random


def custom_train(self):
    total_loss = 0

    total_loader_time = 0
    total_gpu_time = 0
    best_acc = 0

    data_iter = iter(self.train_gen)
    for i in range(self.num_iters):
        self.iter += 1

        start = time.time()

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(self.train_gen)
            batch = next(data_iter)

        total_loader_time += time.time() - start

        start = time.time()
        loss = self.step(batch)
        total_gpu_time += time.time() - start

        total_loss += loss
        self.train_losses.append((self.iter, loss))

        if self.iter % self.print_every == 0:
            info = 'iter: {:06d} - train loss: {:.3f} - lr: {:.2e} - load time: {:.2f} - gpu time: {:.2f}'.format(self.iter,
                    total_loss/self.print_every, self.optimizer.param_groups[0]['lr'],
                    total_loader_time, total_gpu_time)

            total_loss = 0
            total_loader_time = 0
            total_gpu_time = 0
            print(info)
            self.logger.log(info)

        if self.valid_annotation and self.iter % self.valid_every == 0:
            val_loss = self.validate()
            acc_full_seq, acc_per_char = self.precision(self.metrics)

            info = 'iter: {:06d} - valid loss: {:.3f} - acc full seq: {:.4f} - acc per char: {:.4f}'.format(self.iter, val_loss, acc_full_seq, acc_per_char)
            print(info)
            self.logger.log(info)

            if acc_per_char > best_acc:
                self.save_checkpoint(self.export_weights)
                best_acc = acc_per_char

def predict_cus(self, sample=None):
        pred_sents = []
        actual_sents = []
        img_files = []
        
        print("len:",len(self.valid_gen))
        # data_iter = iter(self.valid_gen)
        # batch = next(data_iter)
        # print(batch)
        for batch in self.valid_gen:
            print("ahaihi")
            batch = self.batch_to_device(batch)
            

            translated_sentence, prob = translate(batch['img'], self.model)

            pred_sent = self.vocab.batch_decode(translated_sentence.tolist())
            actual_sent = self.vocab.batch_decode(batch['tgt_output'].tolist())

            img_files.extend(batch['filenames'])

            pred_sents.extend(pred_sent)
            actual_sents.extend(actual_sent)
            
            if sample != None and len(pred_sents) > sample:
                break

        return pred_sents, actual_sents, img_files, prob

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
    parser.add_argument('--pretrained', action='store_true',
                        default=False, help='Train from pretrained model')
    parser.add_argument('--ckpt_path', type=str, default='',
                       help='Path to checkpoint for resume training')
    parser.add_argument('--device', type=str,
                        default='cuda:0', help='Device cuda:0 or cpu')
    return parser.parse_args()

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def main():
    args = get_args()

    with open(args.config, "r") as stream:
        try:
            custom_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        
    print('config: ', custom_config)

    set_seed(custom_config['seed'])

    if custom_config['model'] == "transformer":
        config = Cfg.load_config_from_name("vgg_transformer")
    elif custom_config['model'] == "seq2seq":
        config = Cfg.load_config_from_name("vgg_seq2seq")
    else:
        raise NotImplementedError("model not supported")

    # dataset_params = {
    #     "name": custom_config['dataset_params']['name'],
    #     "data_root": custom_config['dataset_params']['data_root'],
    #     "train_annotation": custom_config['dataset_params']['train_annotation'],
    #     "valid_annotation": custom_config['dataset_params']['valid_annotation'],
    # }

    dataset_params = custom_config['dataset_params']

    save_path = custom_config['save_path']
    export_path = ""
    if save_path is not None:
        export_path = save_path
    else:
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        export_path = f"checkpoints/vietocr/{current_datetime}"
    
    Path(export_path).mkdir(parents=True, exist_ok=True)

    params = {
        "batch_size": custom_config['params']['batch_size'],
        "print_every": custom_config['params']['print_every'],
        "valid_every": custom_config['params']['valid_every'],
        "iters": custom_config['params']['iters_per_epoch'] * custom_config['params']['epochs'],
        "export": f"{export_path}/vgg_{custom_config['model']}.pth",
        # "metrics": 10000,
    }

    config['vocab'] = custom_config['vocab']
    config['backbone'] = custom_config['backbone']
    config['dataloader'] = custom_config['dataloader']
    config["trainer"].update(params)
    config["dataset"].update(dataset_params)
    config["device"] = args.device
    
    # config['transformer']['max_seq_length'] = 150

    if args.pretrained:
        print("Train using pretrained model")

    Trainer.train = custom_train
    # Trainer.predict = predict_cus
    trainer = Trainer(config, pretrained=args.pretrained)
    trainer.config.save(f"{export_path}/config.yml")

    # if args.resume_from is not None:
    #     print(f"Resume from checkpoint {args.resume_from}")
    #     trainer.model.load_state_dict(torch.load(args.resume_from))
    #     print("Evaluating checkpoint weights...")
    #     word_acc, char_acc = trainer.precision()
    #     print(f"Word acc: {word_acc}; char acc: {char_acc}")

    if args.ckpt_path:
        checkpoint = torch.load(args.ckpt_path)
        print(f"Resume from checkpoint {args.ckpt_path}")

        # trainer.optimizer.load_state_dict(checkpoint['optimizer'])
        trainer.model.load_state_dict(checkpoint['state_dict'])
        # trainer.iter = checkpoint['iter']

        # trainer.train_losses = checkpoint['train_losses']

        print("Evaluating checkpoint weights...")
        word_acc, char_acc = trainer.precision()
        print(f"Word acc: {word_acc}; char acc: {char_acc}")

    trainer.train()

if __name__ == "__main__":
    main()
