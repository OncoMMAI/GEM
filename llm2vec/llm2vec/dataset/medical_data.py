import pandas as pd
import random
from typing import Optional
from .dataset import DataSample, TrainSample, Dataset
from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="INFO")

class MedicalData(Dataset):
    """
    Dataset class for unsupervised SimCSE fine-tuning.
    It reads the pre-training corpus where each row is a single document.
    """
    def __init__(
        self,
        dataset_name: str = "medical",
        split: str = "train",
        file_path: Optional[str] = None,
        effective_batch_size: int = 32,
        separator: str = "\n\n",
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.effective_batch_size = effective_batch_size
        self.separator = separator
        self.data = []
        self.load_data(file_path)

    def __len__(self):
        return len(self.data)

    def load_data(self, file_path: Optional[str] = None):
        logger.info(f"Loading Medical data for SimCSE from {file_path}...")

        # The pre-training corpus has a single 'text' column with formatted reports
        df = pd.read_csv(file_path)

        all_samples = []
        for idx, row in df.iterrows():
            instruction = "Encode this medical report for semantic similarity comparison:"
            
            # The 'text' column already contains the cancer type and report
            text_with_instruction = f"{instruction}{self.separator}{row['text']}"

            # For SimCSE, the query and positive are the same text.
            # Negatives are other samples within the same batch.
            sample = DataSample(
                id_=idx,
                query=text_with_instruction,
                positive=text_with_instruction,
                negative=None,
                task_name="medical_similarity",
                label=1.0,
            )
            all_samples.append(sample)

        self.data = all_samples
        logger.info(f"Loaded {len(self.data)} samples for unsupervised training.")

    def __getitem__(self, index):
        sample = self.data[index]
        if self.split == "train":
            # For SimCSE, we just need two copies of the same text
            return TrainSample(
                texts=[sample.query, sample.positive],
                label=sample.label,
            )
        else:
            raise ValueError("MedicalData only supports the 'train' split for SimCSE.")


MEDICAL_PROMPT = "Compare these medical reports for semantic similarity:"

class MedicalPairData(Dataset):
    """
    Dataset class for supervised fine-tuning on matched/unmatched pairs.
    Reads a CSV with 'sentence1', 'sentence2', and 'label' columns.
    """
    def __init__(
        self,
        dataset_name: str = "medical_pairs",
        split: str = "train",
        file_path: Optional[str] = None,
        effective_batch_size: int = 32,
        separator: str = "\n\n",
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.effective_batch_size = effective_batch_size
        self.separator = separator
        self.data = []
        self.load_data(file_path)

    def __len__(self):
        return len(self.data)

    def load_data(self, file_path: Optional[str] = None):
        logger.info(f"Loading Medical Pair data from {file_path} for split '{self.split}'...")
        df = pd.read_csv(file_path)

        if self.split == "train":
            # For MNRL, we only need positive pairs for training.
            num_original_samples = len(df)
            df = df[df["label"] == 1].reset_index(drop=True)
            num_positive_samples = len(df)
            logger.info(
                f"Filtered for positive pairs for training. Kept {num_positive_samples} out of {num_original_samples} total samples."
            )

        all_samples = []
        for idx, row in df.iterrows():
            # Format the input texts with instructions, using the prompt defined above
            query = f"{MEDICAL_PROMPT}{self.separator}Report: {row['sentence1']}"
            document = f"Report: {row['sentence2']}"

            sample = DataSample(
                id_=idx,
                query=query,
                positive=document, # The base class uses 'positive' for the second sentence
                negative=None,
                task_name="medical_similarity",
                label=float(row["label"]),
            )
            all_samples.append(sample)

        self.data = all_samples
        logger.info(f"Loaded {len(self.data)} samples for the '{self.split}' split.")

    def __getitem__(self, index):
        sample = self.data[index]
        return TrainSample(texts=[sample.query, sample.positive], label=sample.label)
