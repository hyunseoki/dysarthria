'''
https://github.com/juice500ml/dysarthria-mtl/blob/main/train.py
'''

import torch
import torch.nn as nn
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel
)
import torch.nn.functional as F


class Wav2Vec2ForClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.cfg = config.task_specific_params
        self.wav2vec2 = Wav2Vec2Model(config)

        self.cls_head = nn.Linear(config.hidden_size, self.cfg['num_classes'])
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.dropout = nn.Dropout(config.final_dropout)


    def forward(self, input_values, input_lengths, cls_labels=None, ctc_labels=None):
        assert ctc_labels.max().item() <= 66, ctc_labels.max().item()

        outputs = self.wav2vec2(
            input_values, 
            attention_mask=input_lengths[:, None],
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        )

        hidden_states = self.dropout(outputs[0]) ## b x t x 1024
        state_lens = self.wav2vec2._get_feat_extract_output_lengths(input_lengths)


        ## clf
        # max_state_len = hidden_states.shape[1] ## t       
        # clf_mask = (torch.arange(max_state_len)[None, :].to(state_lens.device) < state_lens[:, None])[:, :, None]
        # avg_states = torch.sum(hidden_states * clf_mask, dim=1) / torch.sum(clf_mask, dim=1)
        
        # cls_logits = self.cls_head(self.dropout(avg_states))
        # cls_loss = F.cross_entropy(cls_logits, cls_labels)


        ## ctc
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


        return ctc_loss, ctc_logits


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
    sample['cls_labels'] = torch.as_tensor([1,1,1], dtype=int)
    model(**sample)
