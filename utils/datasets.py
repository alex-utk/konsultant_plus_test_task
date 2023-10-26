import torch
import pandas as pd
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """Класс датасета для работы с one-how-encoded векторами"""

    def __init__(self: Dataset, df: pd.DataFrame) -> None:
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = df

    def __len__(self: Dataset) -> int:
        return len(self.data)

    def __getitem__(self: Dataset, idx: int) -> tuple[torch.Tensor, int]:
        row = self.data.iloc[idx]
        features = torch.from_numpy(row['features'].toarray()).to(torch.float32)
        features = torch.squeeze(features)
        label = int(row['label'])
  
        return features, label
    
    
class RuBertDataset(TextDataset):
    """Датасет, раюотающий с эмбеддингами RuBert-tiny-2.

    Отличий минимум, просто немного по другому возвращаются эмбеддинги"""

    def __getitem__(self: TextDataset, idx: int) -> tuple[torch.Tensor, int]:
        row = self.data.iloc[idx]
        features = torch.from_numpy(row['features']).to(torch.float32)
        label = int(row['label'])
     
        return features, label
