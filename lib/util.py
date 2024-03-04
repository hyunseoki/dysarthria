import os
import random
import itertools
import tqdm
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fe = fm.FontEntry(fname=r'/usr/share/fonts/truetype/nanum/NanumGothic.ttf', name='NanumGothic') #파일 저장되어있는 경로와 이름 설정
fm.fontManager.ttflist.insert(0, fe)
plt.rcParams.update({'font.family': 'NanumGothic'})
plt.rcParams['axes.unicode_minus'] = False


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_confusion_matrix(cm, save_fn, metric=None, target_names=None, cmap=None, count=True, title='Confusion matrix'):
    if cmap is None:
        cmap = plt.get_cmap('Blues')
        
    plt.figure(figsize=(16, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=0, fontsize=12.0)
        plt.yticks(tick_marks, target_names, fontsize=12.0)
    
    if count:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if cm[i, j] != 0:
                plt.text(j, i, "{}".format(cm[i, j]),
                        horizontalalignment="center",
                        fontsize=12.0,
                        color="red")

    plt.tight_layout()
    plt.ylabel('True label')

    if metric is None:
        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy
        plt.xlabel('Predicted label\n\naccuracy={:0.3f}; misclass={:0.3f}'.format(accuracy, misclass))
    else:
        plt.xlabel('Predicted label\n\nPrecision={:0.3f}, Recall={:0.3f}, F1_Score={:0.3f}'.format(metric['precision'], metric['recall'], metric['f1_score']))
    
    plt.savefig(save_fn, bbox_inches="tight")


@torch.no_grad()
@torch.cuda.amp.autocast()
def plot_F1score(data_loader, model, device, save_dir):
    data_set = data_loader.dataset
    batch_size = data_loader.batch_size
    results = np.zeros(shape=(len(data_set), 4))
    model.eval()
    model.to(device)

    for idx, sample in enumerate(tqdm.tqdm(data_loader)):
        sample = {k: v.to(device) for k, v in sample.items()}
        ret = model(**sample)
        logits = ret['clf_logits']
        batch_index = idx * batch_size
        results[batch_index:batch_index+batch_size] += logits.clone().detach().cpu().numpy()

    true = [data_set.label2int[label] for label in data_set.df['class']]
    pred = np.array([np.argmax(result) for result in results])
    
    precision, recall, f1_score, _= precision_recall_fscore_support(y_true=true, y_pred=pred, average='macro')
    metric = {'precision': precision, 'recall': recall, 'f1_score': f1_score}

    conf_mat = confusion_matrix(true, pred)
    save_fn = os.path.join(save_dir, 'confusion_matrix.png')
    plot_confusion_matrix(conf_mat, metric=metric, save_fn=save_fn, target_names=list(data_set.label2int.keys()), count=True)

    with open(os.path.join(save_dir, 'test_result.txt'), 'w') as f:
        f.write(f'test acc : {f1_score}') 