import os
import argparse
import datetime
import torch
import glob
import pandas as pd

import transformers
from transformers import Wav2Vec2CTCTokenizer
from lib.engine import BaseTrainer
from lib.util import plot_F1score, seed_everything
from lib.metric import ctc_metric, cls_metric
from lib.dataset import DysarthriaDataset
from model import Wav2Vec2ForClassification
import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--kfold_idx', type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument('--base_path', type=str, default='/home/hyunseoki_rtx3090/ssd1/02_src/speech_recognition/dysarthria_clf/')
    parser.add_argument('--save_path', type=str, default='./checkpoint')

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-5)

    parser.add_argument('--attention', action='store_true')
    parser.add_argument('--mtl', action='store_true', default=True)
    parser.add_argument('--weight', type=float, default=0.5)

    parser.add_argument('--num_heads', type=int, default=4)

    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--comments', type=str, default=None)

    cfg = parser.parse_args()
    if cfg.comments is not None:
        cfg.save_path = os.path.join(cfg.save_path, cfg.comments)
    cfg.save_path = os.path.join(cfg.save_path, datetime.datetime.now().strftime("%m%d%H%M%S")) 

    print('=' * 50)
    print('[info msg] arguments')
    for key, value in vars(cfg).items():
        print(key, ":", value)
    print('=' * 50)

    return cfg


def prepare_model(cfg):
    model_cfg, *_ = transformers.PretrainedConfig.get_config_dict("hyunseoki1/dysarthria")
    model_cfg["task_specific_params"] = {
        "num_classes": 4, 
        "mtl": cfg.mtl,
        "attention": cfg.attention, 
        "num_heads": cfg.num_heads
    }

    model = Wav2Vec2ForClassification.from_pretrained(
        "hyunseoki1/dysarthria",
        config=transformers.Wav2Vec2Config.from_dict(model_cfg),
    )
    model.to(cfg.device)

    return model


def collator(batch):
    return {
        "input_values": torch.nn.utils.rnn.pad_sequence(
                            [torch.FloatTensor(sample['audio']) for sample in batch],
                            batch_first=True,
                            padding_value=0.0,
                        ),
        "input_lengths": torch.LongTensor([sample['audio_len'] for sample in batch]),
        "clf_labels": torch.LongTensor(
                            [sample['clf_label'] for sample in batch]
                        ),
        "ctc_labels": torch.nn.utils.rnn.pad_sequence(
            [torch.IntTensor(sample["ctc_label"]) for sample in batch],
            batch_first=True,
            padding_value=-100,
        ),
    }



def main():
    cfg = parse_args()
    seed_everything()

    df = pd.read_csv(os.path.join(cfg.base_path, 'data', f'fold{cfg.kfold_idx}.csv'))
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("hyunseoki1/dysarthria")
    train_dataset = DysarthriaDataset(base_path=cfg.base_path, df=df[df['phase'] == 'train'], tokenizer=tokenizer)
    valid_dataset = DysarthriaDataset(base_path=cfg.base_path, df=df[df['phase'] == 'test'], tokenizer=tokenizer)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collator,
        shuffle=True,
        pin_memory=True,
        num_workers=cfg.num_workers,
        persistent_workers=True
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collator,
        shuffle=False,
        pin_memory=True,
        num_workers=cfg.num_workers,
        persistent_workers=True
    )

    model = prepare_model(cfg)
    if cfg.resume:
        model.load_state_dict(torch.load(cfg.resume))
    
    
    model.train()
    model.to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98), eps=1e-08)
    metric_func = {'ctc_metric': ctc_metric, 'cls_metric': cls_metric}
    scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer=optimizer, 
            num_warmup_steps=50,
            num_training_steps=cfg.epochs,
            num_cycles=5,
        )
    trainer = BaseTrainer(
        model=model,
        train_loader=train_data_loader,
        valid_loader=valid_data_loader,
        scheduler=scheduler,
        num_epochs=cfg.epochs,
        save_path=cfg.save_path,
        metric_func=metric_func,
        tokenizer=tokenizer,
        optimizer=optimizer,
        device=cfg.device,
        weight=cfg.weight,
    )

    trainer.fit()

    with open(os.path.join(cfg.save_path, 'config.txt'), 'w') as f:
        for key, value in vars(cfg).items():
            f.write('{} : {}\n'.format(key, value)) 

    weight_fn = glob.glob(f'{cfg.save_path}/**/*.pth', recursive=True)[0]
    model.load_state_dict(torch.load(weight_fn))
    plot_F1score(
        data_loader=valid_data_loader,
        model=model,
        device=cfg.device,
        save_dir=cfg.save_path,
    )


if __name__ == '__main__':
    main()