# Lyrics Language Models

Experiments on a song-lyrics corpus: a smoothed trigram language model using RoBERTa tokenization, and a GRU-based neural language model using GPT-2 tokens. Scripts compute checkpoint metrics, perplexities, and generate short lyric continuations.

## Features
- Uses `RobertaTokenizerFast` to explore tokenization of lyrics.
- Trigram LM with add-one smoothing and backoff, reporting next-token probabilities and perplexity for sample lines.
- GPT-2-tokenized dataset chunking, GRU language model training, loss curves, and autoregressive generation.
- Saves intermediate checkpoints and sample perplexities to output files.

## Tech Stack
- Python 3.9+
- PyTorch, Transformers, NumPy, matplotlib, csv/regex stdlib

## Setup
```bash
python -m venv .venv
. .venv/Scripts/activate
pip install torch transformers numpy matplotlib
```
Add the dataset file beside the scripts:
- `songs.csv` (with columns: title, album, lyrics). The scripts skip the header and last 5 rows to mirror the original lab.

GPU is recommended for the GRU training loop but the code will run on CPU (slower).

## Usage
Trigram LM checkpoints:
```bash
python trigram_language_model.py
```
Outputs: next-token probability printouts and perplexities for five sample cases.

GRU language model training & generation:
```bash
python gru_language_model.py
```
Outputs: `a2_p2_Shah_116727594_OUTPUT.txt`, `loss_curve.pdf`, and generated lyric samples printed to console.

## Folder Structure
- `trigram_language_model.py` — tokenizer demo + trigram LM with smoothing
- `gru_language_model.py` — GPT-2 tokenization, GRU LM training, perplexity eval, generation
- *(dataset expected but not committed)*

## Possible Improvements
- Replace Colab `files.upload/download` with local path args and proper error handling.
- Add train/validation split and evaluation loop for the GRU model.
- Enable deterministic training (seeds) and checkpoint saving/loading.
- Package common utilities (tokenization, chunking) into reusable modules; add CLI flags for hyperparameters.
- Provide `requirements.txt` and lightweight tests for chunking and probability calculations.
