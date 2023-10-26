import re
import time
import torch
import numpy as np
from pymystem3 import Mystem
from nltk.corpus import stopwords
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from torchmetrics.classification import Recall, Precision


mystem = Mystem()
allowed_symbols = re.compile(u'[0-9]|[^\u0400-\u04FF ]')
ru_stopwords = stopwords.words("russian")


def tokenize_and_filter(text: str, ru_stopwords: list[str]) -> list[str]:
    """Лемматизация, токенизация и постпроцессинг

    Args:
        text (str): текст
        mystem (Mystem): лемматизатор яндекса
        ru_stopwords (list[str]): стоп слова nltk

    Returns:
        list[str]: очищенный список токенов
    """
    tokens_list = mystem.lemmatize(text) # токенизация и лемматизация лемматизатором яндекса
    tokens_list = [token.strip() for token in tokens_list] # очищаем от лишних пробелов
    # так как у нас правило token == 'не' идет самое первое, то за счет оптимизации
    # логических операций слово не будет даже если оно есть в списке stopwords
    cleared_tokens = list(filter(lambda token: token == 'не' or (token not in ru_stopwords and len(token) > 2),
                          tokens_list))
    
    cleared_text = ' '.join(cleared_tokens)
    return cleared_text


def preprocess_text(text: str) -> str:
    """Препроцессинг текста

    Args:
        text (str): текст

    Returns:
        str: обработанный текст
    """
    text = allowed_symbols.sub(' ', text).lower()
    text = tokenize_and_filter(text, ru_stopwords)
    return text


def train_model(
    model: nn.Module,
    train_loader: nn.CrossEntropyLoss,
    valid_loader: DataLoader,
    criterion: torch.nn.CrossEntropyLoss,
    sheduler: torch.optim.lr_scheduler,
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    filename: str,
    patience: int = 5
) -> None:
    """Запуск обучения модели

    Args:
        model (nn.Module): модель
        train_loader (nn.CrossEntropyLoss): loader с тестовыми данными
        valid_loader (DataLoader): loader с валидационными данными
        criterion (torch.nn.CrossEntropyLoss): loss
        sheduler (torch.optim.lr_scheduler): sheduler
        optimizer (torch.optim.Optimizer): optimizer
        n_epochs (int): количество эпох
        filename (str): куда сохранять веса модели
        patience (int, optional): сколько эпох лосс можкет расти перед остановокой обучения. Defaults to 5.
    """
    model.train()
    model.cuda()
    valid_loss_min = np.Inf
    current_p = 0

    for epoch in range(1, n_epochs + 1):
        print(time.ctime(), 'Epoch:', epoch)

        # train
        train_loss = []
        for data, target in train_loader:
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        val_loss = []
        for data, target in valid_loader:
            data, target = data.cuda(), target.cuda()
            with torch.inference_mode():
                output = model(data)
            loss = criterion(output, target)
            val_loss.append(loss.item())

        valid_loss = np.mean(val_loss)
        print(f'Epoch {epoch}, train loss: {np.mean(train_loss):.4f}, valid loss: {valid_loss:.4f}.')

        sheduler.step(valid_loss)
        # если лосс стал меньше, то сохраняем чекпоинт, а терпение сбрасываем
        if valid_loss <= valid_loss_min:
            print(f'Validation loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}). Saving model.')
            torch.save(model.state_dict(), filename)
            valid_loss_min = valid_loss
            current_p = 0
        # если лосс стал больше, то терпение нарастает
        else:
            current_p += 1
            print(f'{current_p} epochs of increasing val loss')
            if current_p > patience:
                print('Stopping training')
                break

  
def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    n_classes: int
) -> tuple[float, float]:
    """_summary_

    Args:
        model (nn.Module): модель
        test_loader (DataLoader): загрузчик тестовых данных
        n_classes (int): количество классов

    Returns:
        tuple(float, float): значения precision, recall
    """
    recall = Recall(task="multiclass", average='micro', num_classes=n_classes)
    precision = Precision(task="multiclass", average='micro', num_classes=n_classes)

    targets = []
    preds = []

    for data, label in test_loader:
        data = data.cuda()
        with torch.inference_mode():
            output = model(data)
            pred = torch.argmax(softmax(output, dim=1))
            pred = pred.cpu()

        targets.append(label.item())
        preds.append(pred)

    targets = torch.tensor(targets, dtype=torch.int)
    preds = torch.tensor(preds, dtype=torch.int)

    precision_score = precision(preds, targets).item()
    recall_score = recall(preds, targets).item()

    return precision_score, recall_score
