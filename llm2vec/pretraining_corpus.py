"""
This script creates a pre-training corpus from raw MRI and Pathology reports
for the Masked Next Token Prediction (MNTP) step of LLM2Vec.
"""
import re
import pandas as pd
from tqdm import tqdm
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split

# --- Text Cleaning Function ---
def clean_text(text):
    """Cleans medical text while preserving numbers, decimals, clinical measurements, and spacing."""
    if not isinstance(text, str):
        return ""

    # Basic normalization
    text = text.lower()

    # Remove URLs, emails
    text = re.sub(r"(?:https?://|www\.)\S+", "", text)
    text = re.sub(r"\b[\w.-]+@[\w.-]+\.\w+\b", "", text)

    phone_patterns = [
        r"\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",  # (415) 476-5236 or 415-476-5236
        r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",  # 4154765236
    ]
    for pattern in phone_patterns:
        text = re.sub(pattern, "", text)

    # Remove fax/tel prefixes and postal codes
    text = re.sub(r"\b(?:tel|fax|telephone):\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b", "", text, flags=re.IGNORECASE)

    # Remove extremely long words
    text = re.sub(r"\b[a-zA-Z]{30,}\b", "", text)

    # Remove extra whitespace and asterisks
    text = re.sub(r"\s+", " ", text)
    text = text.replace("*", "")
    text = text.replace("/", "")

    return text.strip()

# --- Data Loading and Combining Functions ---
COLUMNS_NEEDED = [
    "Patient MRN",
    "Report Text",
    "Accession Number",
    "Exam Completed Date",
    "Patient Sex",
    "Patient Age",
    "Patient Status",
]

CANCER_FILES = [
    "search_astrocytoma_MR_perReport.xlsx",
    "search_Ependymoma_MR_perReport.xlsx",
    "search_Ganglioglioma_MR_perReport.xlsx",
    "search_glioblastoma_MR_perReport.xlsx",
    "search_Gliomatosis_MR_perReport.xlsx",
    "search_Gliosarcoma_MR_perReport.xlsx",
    "search_oligoastrocytoma_MR_perReport.xlsx",
    "search_oligodendroglioma_MR_perReport.xlsx",
    "search_Xanthoastrocytoma_MR_perReport.xlsx",
]

def get_cancer_type(filename: str) -> str:
    """Extract and standardize cancer type from filename."""
    cancer_type = filename.replace("search_", "").split("_")[0]
    return cancer_type.lower()

def load_all_reports(base_path: str) -> pd.DataFrame:
    """Load and combine all cancer type data files from both sources."""
    all_dfs = []

    for filename in tqdm(CANCER_FILES, desc="Loading raw report files"):
        try:
            cancer_type = get_cancer_type(filename)

            # Load MRI data
            mri_file = f"{base_path}/{filename}"
            mri_df = pd.read_excel(mri_file, usecols=COLUMNS_NEEDED)
            mri_df["cancer_type"] = cancer_type
            all_dfs.append(mri_df)

            # Load pathology data
            path_filename = filename.replace("_MR_", "_pathology_")
            path_file = f"{base_path}/{path_filename}"
            path_df = pd.read_excel(path_file, usecols=COLUMNS_NEEDED)
            path_df["cancer_type"] = cancer_type
            all_dfs.append(path_df)

        except Exception as e:
            logging.error(f"Error loading {filename}: {str(e)}")
            continue

    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df

# --- Main Script Logic ---
def create_pretraining_corpus(raw_data_path: str, output_path: str):
    """
    Creates and saves the pre-training corpus for MNTP, splitting it into
    training and evaluation sets.
    """
    print("Loading and combining raw data...")
    all_reports_df = load_all_reports(raw_data_path)
    all_reports_df.columns = [col.lower().replace(" ", "_") for col in all_reports_df.columns]

    print(f"Loaded a total of {len(all_reports_df)} reports.")

    print("Cleaning report text...")
    all_reports_df["clean_text"] = all_reports_df["report_text"].apply(clean_text)
    
    # Drop reports that are empty after cleaning
    original_count = len(all_reports_df)
    print(f"Before cleaning: {len(all_reports_df)} reports")
    all_reports_df.dropna(subset=['clean_text'], inplace=True)
    all_reports_df = all_reports_df[all_reports_df['clean_text'] != '']

    print(f"After cleaning: {len(all_reports_df)} reports")

    print(f"Empty reports removed: {original_count - len(all_reports_df)}")
    
    print("Formatting text for MNTP...")
    all_reports_df["formatted_text"] = all_reports_df.apply(
        lambda row: f"Cancer type: {row['cancer_type']}\n\nReport: {row['clean_text']}",
        axis=1
    )
    
    # The MNTP training script will likely need a column named 'text'
    corpus_df = all_reports_df[["formatted_text"]].rename(columns={"formatted_text": "text"})
    
    print("Splitting data into training and evaluation sets (90/10 split)...")
    train_df, eval_df = train_test_split(corpus_df, test_size=0.1, random_state=42)
    
    # Ensure output directory exists
    output_p = Path(output_path)
    output_dir = output_p.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define output paths
    output_stem = output_p.stem
    train_output_path = output_dir / f"{output_stem}_train.csv"
    eval_output_path = output_dir / f"{output_stem}_eval.csv"

    # Save to separate CSV files
    train_df.to_csv(train_output_path, index=False)
    eval_df.to_csv(eval_output_path, index=False)

    print(f"\nSuccessfully created training corpus with {len(train_df)} documents.")
    print(f"Saved to: {train_output_path}")
    print(f"Successfully created evaluation corpus with {len(eval_df)} documents.")
    print(f"Saved to: {eval_output_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Path from your 01_create_dataset.ipynb notebook
    RAW_DATA_PATH = "/data/users/neob/mPower/perReport_12122024"
    
    # Suggested output path for the pre-training corpus
    OUTPUT_CORPUS_PATH = "data/pretraining/mntp_corpus.csv"
    
    create_pretraining_corpus(RAW_DATA_PATH, OUTPUT_CORPUS_PATH)
