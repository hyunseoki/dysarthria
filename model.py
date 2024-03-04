'''
https://github.com/juice500ml/dysarthria-mtl/blob/main/train.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel
)


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_rate=0.1):
        super(SelfAttention, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, mask=None):
        x = x.permute(1, 0, 2)  # Change the shape to (sequence_length, batch_size, hidden_size)
        x, attention = self.multihead_attention(x, x, x, key_padding_mask=mask)
        x = x.permute(1, 0, 2)  # Change the shape back to (batch_size, sequence_length, hidden_size)
        x = self.layer_norm(x)
        return x, attention


class Wav2Vec2ForClassification(Wav2Vec2PreTrainedModel):
    MAX_LEN = 480_000 ## 30 sec

    def __init__(self, config):
        super().__init__(config)
        self.cfg = config.task_specific_params
        self.num_classes = self.cfg['num_classes']
        self.wav2vec2 = Wav2Vec2Model(config)
        # self.wav2vec2.freeze_feature_encoder()
        self.cls_head = nn.Linear(config.hidden_size, 4)
        self.dropout = nn.Dropout(config.final_dropout)
        
        if self.cfg['mtl']:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        if self.cfg['attention']:
            self.pos_embedding = nn.Embedding(self.MAX_LEN, config.hidden_size)
            self.attention = SelfAttention(config.hidden_size, num_heads=self.cfg['num_heads'])

    def attention_forward(self, hidden_states, mask=None): # b x t x 1024
        batch_size, seq_len, _ = hidden_states.shape
        pos = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1).to(hidden_states.device)
        pos = self.pos_embedding(pos)
        ret, attention = self.attention(hidden_states+pos, mask[:, :, 0])

        return ret, attention


    def forward(self, input_values, input_lengths, clf_labels, ctc_labels):
        outputs = self.wav2vec2(
            input_values, 
            attention_mask=input_lengths[:, None],
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        )

        hidden_states = outputs[0] ## b x t x 1024
        hidden_states = self.dropout(hidden_states)

        ## clf
        max_state_len = hidden_states.shape[1] ## t
        state_lens = self.wav2vec2._get_feat_extract_output_lengths(input_lengths)
        mask = (torch.arange(max_state_len)[None, :].to(state_lens.device) < state_lens[:, None])[:, :, None]

        if self.cfg['attention']:
            avg_states, _ = self.attention_forward(hidden_states, mask)

        avg_states = torch.sum(hidden_states * mask, dim=1) / torch.sum(mask, dim=1)
        clf_logits = self.cls_head(avg_states)
        clf_loss = F.cross_entropy(clf_logits, clf_labels)

        # ctc
        if self.cfg['mtl']:
            labels_mask = ctc_labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = ctc_labels.masked_select(labels_mask)
            ctc_logits = self.lm_head(hidden_states)
            log_probs = nn.functional.log_softmax(ctc_logits, dim=-1, dtype=torch.float32).transpose(0, 1)
            with torch.backends.cudnn.flags(enabled=False):
                ctc_loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    state_lens,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

            return {
                'clf_loss': clf_loss,
                'clf_logits': clf_logits, 
                'ctc_loss': ctc_loss, 
                'ctc_logits': ctc_logits
            }
        else:
            return {'clf_loss': clf_loss, 'clf_logits': clf_logits}


if __name__ == '__main__':
    import transformers, torch
    model_cfg, *_ = transformers.PretrainedConfig.get_config_dict("facebook/wav2vec2-xls-r-300m")
    model_cfg["gradient_checkpointing"] = True
    model_cfg["task_specific_params"] = {"num_classes": 4, "attention":True}
    model = Wav2Vec2ForClassification.from_pretrained(
        "facebook/wav2vec2-xls-r-300m",
        config=transformers.Wav2Vec2Config.from_dict(model_cfg),
    )

    sample = {}
    sample['input_values'] = torch.randn((3, 450_000))
    sample['input_lengths'] = torch.randint(high=450_000, size=(3,))
    sample['clf_labels'] = torch.as_tensor([1,1,1], dtype=int)
    model(**sample)
