import torch
import torch.nn as nn
import torch.nn.functional as F


class BagOfWordsModel(nn.Module):
    """Модель классификатора Bag-of-Words"""

    def __init__(self: nn.Module, embed_dim: int, num_class: int) -> None:
        super(BagOfWordsModel, self).__init__()

        self.fc1 = nn.Linear(embed_dim, 300)
        self.bn1 = nn.BatchNorm1d(300)
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(300, 200)
        self.bn2 = nn.BatchNorm1d(200)
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(200, 100)
        self.bn3 = nn.BatchNorm1d(100)
        self.dropout3 = nn.Dropout(p=0.2)
        self.fc4 = nn.Linear(100, num_class)

        self.init_weights()


    def init_weights(self: nn.Module) -> None:
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)

        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
        nn.init.zeros_(self.fc4.bias)


    def forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout2(x)

        x = F.relu(self.fc3(x))
        x = self.bn3(x)
        x = self.dropout3(x)

        x = self.fc4(x)

        return x


class RuBertClassifier(nn.Module):
    """Модель классификатора для эмбеддингов RuBert-tiny-2"""
    
    def __init__(self: nn.Module, embed_dim: int, num_class: int) -> None:
        super(RuBertClassifier, self).__init__()
        self.fc1 = nn.Linear(embed_dim, num_class)  # 312 - SentenceTransformer output
        
        self.init_weights()

    def init_weights(self: nn.Module) -> None:
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

    def forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)

        return x
