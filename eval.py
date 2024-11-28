import torch

from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def compute_cider(predictions, references):
    """ 
    Computes CIDEr score. 
    `predictions` and `references` should be lists of generated and ground truth text for each sample. 
    """

    cider_scorer = Cider()
    references_dict = {idx: [text] for idx, text in enumerate(references)}
    predictions_dict = {idx: [text] for idx, text in enumerate(predictions)}
    cider_score, _ = cider_scorer.compute_score(
        references_dict, predictions_dict)

    return cider_score


def compute_accuracy(correct, total):
    """
    Compute the accuracy of the model's predictions.

    Parameters:
    - correct: the number of correct prediction
    - total: the number of total smaples

    Returns:
    - accuracy: float, the accuracy of predictions
    """

    return correct / total


def evaluate(model, val_loader, criterion, idx2label, dataset_name):
    model.eval()
    total_correct = 0
    total_loss = 0
    total_samples = 0
    all_predictions = []
    all_references = []

    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            text_inputs_lst = batch.pop('text_inputs_lst')
            img_inputs_lst = batch.pop('img_inputs')
            labels = batch.pop('labels')

            text_inputs_lst = [
                {k: v.squeeze().to(device, non_blocking=True)
                 for k, v in input_ids.items()}
                for input_ids in text_inputs_lst]
            img_inputs_lst = [inputs.to(device, non_blocking=True)
                              for inputs in img_inputs_lst]
            labels = labels.to(device, non_blocking=True)

            logits = model(text_inputs_lst, img_inputs_lst)

            loss = criterion(logits, labels)

            _, preds = torch.max(logits, 1)

            total_batch_samples = labels.size(0)
            batch_loss_sum = loss.item() * total_batch_samples

            total_samples += total_batch_samples
            total_loss += batch_loss_sum

            if dataset_name == 'openvivqa':
                pred_texts = [idx2label[pred.item()] for pred in preds]
                label_texts = [idx2label[label.item()] for label in labels]

                all_predictions += pred_texts
                all_references += label_texts
            else:
                correct = (preds == labels).sum().item()
                total_correct += correct

    eval_loss = total_loss / total_samples

    if dataset_name == 'openvivqa':
        eval_cider = compute_cider(all_predictions, all_references)
        
        bleu_scorer = Bleu(4)
        meteor_scorer = Meteor()
        rouge_scorer = Rouge()

        references_dict = {idx: [text] for idx, text in enumerate(all_references)}
        predictions_dict = {idx: [text] for idx, text in enumerate(all_predictions)}

        eval_bleu, _ = bleu_scorer.compute_score(references_dict, predictions_dict)
        eval_meteor, _ = meteor_scorer.compute_score(references_dict, predictions_dict)
        eval_rouge, _ = rouge_scorer.compute_score(references_dict, predictions_dict)

        return {
            'val_loss': eval_loss,
            'val_acc': -1,
            'val_cider': eval_cider,
            'val_bleu_1': eval_bleu[0],
            'val_bleu_2': eval_bleu[1],
            'val_bleu_3': eval_bleu[2],
            'val_bleu_4': eval_bleu[3],
            'val_meteor': eval_meteor,
            'val_rouge': eval_rouge
        }
    else:
        eval_acc = compute_accuracy(total_correct, total_samples)
        return {
            'val_loss': eval_loss,
            'val_acc': eval_acc,
            'val_cider': -1,
            'val_bleu_1': -1,
            'val_bleu_2': -1,
            'val_bleu_3': -1,
            'val_bleu_4': -1,
            'val_meteor': -1,
            'val_rouge': -1
        }
