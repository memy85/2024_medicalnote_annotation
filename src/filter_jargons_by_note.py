"""Produce a fine-tuning dataset by keeping only jargon phrases that appear in their notes."""

from argparse import ArgumentParser
from pathlib import Path
import json
import random

from pandas import DataFrame
import pandas as pd
import re

from utils import PROJECT_PATH, load_config, load_cv_dataset

config = load_config()

JARGON_LINE_RE = re.compile(r"^\s*(\d+)[\.\)]\s*(.+)$")
DATA_PATH = PROJECT_PATH.joinpath("data/processed")

def load_dataset() -> DataFrame : 
    path = DATA_PATH.joinpath("discharge_dataset_augmented_10000.json")
    df = pd.read_json(path)
    return df

def load_mad10k() -> DataFrame : 
    path = DATA_PATH.joinpath("mad10k_raw.json")
    df = pd.read_json(path, lines=True, orient="records")
    return df


def _term_in_note(term: str, note_text: str) -> bool:
    if not term or not note_text:
        return False
    return term.lower() in note_text.lower()

def _parse_ranked_terms(raw_output: str) -> list[tuple[int, str]]:
    """Return (rank, term) pairs from a numbered raw output block."""
    if not raw_output:
        return []
    parsed = []
    for line in raw_output.splitlines():
        match = JARGON_LINE_RE.match(line.strip())
        if not match:
            continue
        rank, term = match.groups()
        parsed.append((int(rank), term.strip()))
    return parsed

def filter_jargons(mad10k_raw : DataFrame) -> pd.DataFrame:
    # note_lookup = notes.set_index("noteid")["text"].to_dict()
    raw_jargons = mad10k_raw["raw_output"]

    # keep_indices = []
    kept_rows = []
    for idx, row in mad10k_raw.iterrows():
        note_text = row["input"]
        # note_text = note_lookup.get(row["noteid"], "")
        # break down jargons
        entries = _parse_ranked_terms(row.get("raw_output", ""))
        matched = sorted(
            ((rank, term)
            for rank, term in entries
            if _term_in_note(term, note_text)
             ),key=lambda pair: pair[0],
            )
        
        if not matched:
            continue
        kept_rows.append(
            {
                "input": note_text,
                "filtered_output": "".join(f"{rank}. {term}\n" for rank, term in matched),
                "ranked_terms": [{"rank": rank, "term": term} for rank, term in matched],
                "raw_output": row.get("raw_output"),
            }
        )
    return pd.DataFrame(kept_rows)


def _format_group(df: pd.DataFrame) -> str:
    lines = []
    for _, row in df.iterrows():
        rank = row.get("ranking")
        if pd.isna(rank):
            lines.append(row["jargon"])
            continue

        if isinstance(rank, float) and rank.is_integer():
            rank_str = str(int(rank))
        else:
            rank_str = str(rank)

        lines.append(f"{rank_str}. {row['jargon']}")
    return "\n".join(lines)


def build_training_set(filtered: pd.DataFrame, prompt_type : str) -> pd.DataFrame:
    template = config.template(prompt_type, "zeroshot")
    train_set = []
    for idx, row in filtered.iterrows() : 
        note = row["input"]
        input_prompt = template.format(note=note)

        output = row["filtered_output"]
        train_set.append({"input" : input_prompt, "output" : output})

    return train_set


def main():
    parser = ArgumentParser(description="Keep only jargons present in their medical note.")
    parser.add_argument(
        "--output",
        "-o",
        default=DATA_PATH.joinpath("final_jargons_training.jsonl"),
        help="Destination path for the fine-tuning dataset.",
    )
    args = parser.parse_args()

    # _, final_jargons, final_notes = load_cv_dataset()
    dataset = load_dataset()
    mad10k_raw = load_mad10k()
    filtered = filter_jargons(mad10k_raw)
    
    for prompt_type in ["generic", "structured"] : 
        training_set = build_training_set(filtered, prompt_type)

        for data_size in [10,100,1000,10000]: 
            random.seed(42)
            if data_size > len(training_set) : 
                sampled_set = training_set
            else : 
                sampled_set = random.sample(training_set, data_size)
        
            with open(DATA_PATH.joinpath(f"mad{data_size}_{prompt_type}.json"), "w") as f :
                json.dump(sampled_set, f)



if __name__ == "__main__":
    main()
