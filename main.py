import os
import glob
import argparse
import datetime
import sys
import torch
import pandas as pd
import tqdm
import transformers
from transformers import Wav2Vec2CTCTokenizer
from lib.util import plot_F1score, seed_everything
from lib.dataset import DysarthriaDataset
from lib.engine import BaseTrainer, AverageMeter
from model import Wav2Vec2ForClassification

import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--kfold_idx', type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument('--base_path', type=str, default='/workspace/01_dataset/aihub/구음장애/dysarthria/')
    parser.add_argument('--save_path', type=str, default='./checkpoint')

    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)

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


def prepare_model(cfg, tokenizer):
    model_cfg, *_ = transformers.PretrainedConfig.get_config_dict("facebook/wav2vec2-xls-r-300m")
    model_cfg["gradient_checkpointing"] = True
    model_cfg["ctc_loss_reduction"] = 'mean'
    # model_cfg["ctc_zero_infinity"] = True
    model_cfg["task_specific_params"] = {"num_classes": 4,}
    model_cfg["vocab_size"] = len(tokenizer)
    model = Wav2Vec2ForClassification.from_pretrained(
        "facebook/wav2vec2-xls-r-300m",
        config=transformers.Wav2Vec2Config.from_dict(model_cfg),
    )
    model.to(cfg.device)

    return model


def get_tokenizer(vocab_fn='./vocab.json'):
    return Wav2Vec2CTCTokenizer(
        vocab_fn,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token=" ",
    )



def collator(batch):
    return {
        "input_values": torch.nn.utils.rnn.pad_sequence(
                            [torch.FloatTensor(sample['audio']) for sample in batch],
                            batch_first=True,
                            padding_value=0.0,
                        ),
        "input_lengths": torch.LongTensor([sample['audio_len'] for sample in batch]),
        "cls_labels": torch.LongTensor(
                            [sample['cls_label'] for sample in batch]
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

    df = pd.read_csv(os.path.join(cfg.base_path, 'data',f'fold{cfg.kfold_idx}.csv'))
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./tokenizer")
    # tokenizer = get_tokenizer()
    # tokenizer.save_pretrained("./tokenizer/")
    train_dataset = DysarthriaDataset(base_path=cfg.base_path, df=df[df['phase'] == 'train'], tokenizer=tokenizer)
    # # test_dataset = DysarthriaDataset(df=df[df['phase'] == 'test'])

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collator,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
        persistent_workers=True
    )

    model = prepare_model(cfg, tokenizer)
    if cfg.resume:
        model.load_state_dict(torch.load(cfg.resume))

    model.train()
    model.to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98), eps=1e-08)

    torch.cuda.empty_cache()
    for epoch in range(cfg.epochs):
        train_loop = tqdm.tqdm(enumerate(train_data_loader), total=len(train_data_loader), desc=f"Epoch [{epoch}]"),

        for step, sample in train_loop:
            sample = {k: v.to(model.device) for k, v in sample.items()}
            ctc_loss, ctc_logits = model(**sample)

            losses = {"ctc_loss": ctc_loss.item()}
            train_loop.set_postfix(losses)
            
            ctc_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
    # for epoch in range(cfg.epochs):
    #     train_loop = tqdm.tqdm(enumerate(train_data_loader))

    #     for step, sample in train_loop:
    #         sample = {k: v.to(model.device) for k, v in sample.items()}
    #         ctc_loss, ctc_logits = model(**sample)

    #         losses = {"ctc_loss": ctc_loss.item()}
    #         train_loop.set_description(
    #             " | ".join([f"Epoch [{epoch}] "] + [f"{k} {v:.4f}" for k, v in losses.items()])
    #         )

    #         ctc_loss.backward()
    #         optimizer.step()
    #         optimizer.zero_grad()

if __name__ == '__main__':
    main()