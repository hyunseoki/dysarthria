import torch
import librosa
import os


class DysarthriaDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, df, tokenizer):
        self.base_path = base_path
        self.df = df
        self.tokenizer = tokenizer

        self.label2int = {'정상': 0, '뇌신경장애': 1, '언어청각장애': 2, '후두장애': 3}
        self.int2label = ['정상', '뇌신경장애', '언어청각장애', '후두장애']


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        wav_fn = item['file_name']
        wav_fn = os.path.join(self.base_path, wav_fn)
        assert os.path.isfile(wav_fn), wav_fn

        audio, sr = librosa.load(wav_fn, sr=16_000)
        audio_len = len(audio)

        cls_label = self.label2int[wav_fn.split('/')[-2]]
        ctc_label = self.tokenizer.encode(item['transcription'])
        ret = {'audio': audio, 
               'audio_len': audio_len,
               'ctc_label': ctc_label,
               'cls_label': cls_label,
               }

        return ret


if __name__ == '__main__':
    import pandas as pd
    from transformers import Wav2Vec2CTCTokenizer

    df = pd.read_csv('/workspace/01_dataset/aihub/구음장애/dysarthria/data/metadata.csv')
    dataset = DysarthriaDataset(base_path='/workspace/01_dataset/aihub/구음장애/dysarthria/', df=df, task='sr')
    dataset[9]
