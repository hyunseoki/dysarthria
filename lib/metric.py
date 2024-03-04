from sklearn.metrics import f1_score
import evaluate
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")


def cls_metric(real, pred):
    score = f1_score(real, pred, average='macro')
    return score


def ctc_metric(true, pred, tokenizer):
    pred = pred.argmax(axis=-1)
    true[true==-100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred)
    true_str = tokenizer.batch_decode(true)

    wer = wer_metric.compute(predictions=pred_str, references=true_str)
    cer = cer_metric.compute(predictions=pred_str, references=true_str)
    
    return wer, cer