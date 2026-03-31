# 🎙️ Hindi ASR System — Fine-Tuning, Cleanup & Lattice Evaluation

> End-to-end fine-tuning and evaluation of `openai/whisper-small` on Hindi speech data sourced from Josh Talks recordings.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Tasks Covered](#tasks-covered)
- [Notebook Structure](#notebook-structure)
- [Prerequisites](#prerequisites)
- [Setup & Usage](#setup--usage)
- [Pipeline Architecture](#pipeline-architecture)
- [Results](#results)
- [File Outputs](#file-outputs)

---

## Overview

This notebook implements a complete Hindi Automatic Speech Recognition (ASR) pipeline built on top of OpenAI's Whisper-small model. The work covers three core tasks:

1. **Data pipeline** — downloading, cleaning, segmenting, and packaging Hindi audio + transcription pairs from Josh Talks recordings into a HuggingFace-compatible dataset.
2. **ASR output cleanup** — a two-stage post-processing pipeline that normalises Hindi number words and tags English loanwords in ASR outputs.
3. **Hindi spelling classification** — a rule-based classifier that labels a 177K-word list as *correct* or *incorrect* based on Devanagari orthography rules.

Model performance is measured using **Word Error Rate (WER)** on both an internal validation set and the FLEURS Hindi test set.

---

## Tasks Covered

| Task | Description |
|------|-------------|
| **Task 1 — Data Pipeline** | Download audio + transcription JSON from Josh Talks URLs, clean to 16kHz mono, trim silence, filter bad segments, build HuggingFace Dataset |
| **Task 2 — ASR Output Cleanup** | Number normalisation (Hindi digit words → Arabic numerals) + English loanword detection in Devanagari and Roman script |
| **Task 3 — Hindi Spelling Classification** | Rule-based Devanagari orthography checker classifying 177K words with confidence levels (high / medium / low) |
| **Evaluation** | WER comparison — baseline Whisper-small vs. fine-tuned model on FLEURS Hindi test set and internal val set |

---

## Notebook Structure

```
Section 1  — Environment Setup
Section 2  — Google Drive & File Upload
Section 3  — Q1: Data Pipeline
  ├── 3.1  Quick URL Sanity Check
  └── 3.2  Full Data Pipeline
Section 4  — Dataset Preparation (HuggingFace Dataset + train/val split)
Section 5  — Model & Training Components
  ├── 5.1  Processor
  ├── 5.2  Feature Extraction & Tokenisation
  ├── 5.3  Data Collator
  ├── 5.4  WER Metric
  └── 5.5  Load Whisper-small
Section 6  — Training
  ├── 6.1  Final Training Run (memory-optimised, grad checkpointing)
  ├── 6.2  Free GPU Memory Between Runs
  └── 6.3  Reference: Earlier Training Configs
Section 7  — Evaluation
  ├── 7.1  Inspect Saved Checkpoints
  ├── 7.2  WER Evaluation on Internal Val Set (pipeline API)
  └── 7.3  WER Evaluation on FLEURS Hindi Test Set
Section 8  — Error Analysis
  ├── 8.1  Stratified 25 Error Samples (low / medium / high WER)
  ├── 8.2  Display All 25 Error Samples
  ├── 8.3  Inspect Available Drive Files
  └── 8.4  Hallucination Fix Demo (no_repeat_ngram_size + repetition_penalty)
Section 9  ASR Output Cleanup Pipeline
  ├── 9.1  Generate Raw ASR Outputs (50-sample subset)
  ├── 9.2  Explore Number Words & English Words in References
  ├── 9.3  Number Normalisation
  ├── 9.4  English Word Detection & Full Pipeline Output
  ├── 9.5  Analyse Pipeline Results
  └── 9.6  Manual Annotation of Missed English Words
Section 10 — Hindi Spelling Classification
  ├── 10.1 Load Word List (177K words from Google Sheets)
  ├── 10.2 Spelling Classifier
  ├── 10.3 Review Low-Confidence Words
  ├── 10.4 Manual Assessment of Low-Confidence Labels
  └── 10.5 High & Medium Confidence Incorrect Words Breakdown
Section 11 — Submission
  ├── 11.1 Export CSV
  ├── 11.2 Download File
```

---

## Prerequisites

### Python Libraries

```bash
pip install transformers datasets librosa soundfile jiwer pandas torch torchaudio
pip install pyenchant indic-nlp-library   # for spelling classifier
```

### Runtime

- **Recommended**: Google Colab with GPU (T4 or better)
- The notebook uses `batch_size=4` with **gradient checkpointing** enabled for memory efficiency
- Google Drive is used for persistent storage of checkpoints and output CSVs

### Data Requirements

- Josh Talks recordings metadata (Excel file with URLs)
- Audio files and transcription JSONs accessible via the provided URLs
- Google Drive path: `/content/drive/MyDrive/whisper_hindi/`

---

## Setup & Usage

1. **Open in Google Colab** — Upload the `.ipynb` file or open it directly from Google Drive.
2. **Mount Google Drive** — Run Section 2 to mount Drive and set up the output directory.
3. **Run Environment Setup** — Section 1 installs all required packages.
4. **Execute sections in order**, 1 → 11. Each section builds on the outputs of the previous one.

> ⚠️ **Note:** The full training run (Section 6.1) can take several hours on a T4 GPU. Earlier training configs are preserved in Section 6.3 for reference.

---

## Pipeline Architecture

```
Metadata Excel (recording URLs)
        ↓
Download audio + transcription JSON for each recording
        ↓
Clean audio  →  16kHz mono, trim silence
        ↓
Clean transcript  →  strip noise, keep Devanagari
        ↓
Filter bad segments (too short / too long / empty)
        ↓
Build HuggingFace Dataset
        ↓
Fine-tune Whisper-small
        ↓
Evaluate on FLEURS Hindi test set  →  report WER
```

### ASR Cleanup Pipeline

```
Raw Whisper ASR Output
        ↓
Stage 1: Number Normalisation
  (तीन सौ चौवन  →  354, with idiom guard to avoid false positives)
        ↓
Stage 2: English Loanword Detection
  (Devanagari loanwords like कॉलेज + Roman script words)
        ↓
Cleaned & Tagged Output
```

### Spelling Classifier Rules (applied in order)

| Priority | Rule | Label |
|----------|------|-------|
| 1 | Roman script word | `incorrect spelling` |
| 2 | Mixed Devanagari + Roman | `incorrect spelling` |
| 3 | Invalid matra sequences (double matra, matra at start, double halant) | `incorrect spelling` |
| 4 | Misplaced nukta | `incorrect spelling` |
| 5 | Invalid chandrabindu placement | `incorrect spelling` |
| 6 | High-frequency known-correct word | `correct spelling` (high confidence) |
| 7 | Valid Devanagari structure | `correct spelling` (medium confidence) |

---

## Results

| Evaluation Set | Model | WER |
|----------------|-------|-----|
| Internal Val Set | Baseline `whisper-small` | *(see notebook output)* |
| Internal Val Set | Fine-tuned model | *(see notebook output)* |
| FLEURS Hindi Test | Baseline `whisper-small` | *(see Section 7.3)* |
| FLEURS Hindi Test | Fine-tuned model | *(see Section 7.3)* |

**Error analysis** (Section 8): 25 error samples were stratified across low / medium / high WER buckets. A hallucination fix using `no_repeat_ngram_size` and `repetition_penalty` was demonstrated on the worst-performing sample.

---

## File Outputs

All outputs are saved to `/content/drive/MyDrive/whisper_hindi/` on Google Drive:

| File | Description |
|------|-------------|
| `dataset_manifest.csv` | Full manifest of cleaned audio segments with paths & transcriptions |
| `error_samples.csv` | 25 stratified WER error samples for analysis |
| `q2_raw_asr.csv` | Raw Whisper ASR outputs on 50-sample subset |
| `q2_pipeline_output.csv` | Cleaned & tagged outputs after number norm + English detection |
| `q3_classified_words.csv` | Full 177K-word classification with labels & confidence |
| `q3_submission.csv` | Final submission file (`word`, `label` columns only) |
| `checkpoints/` | Fine-tuned model checkpoints saved during training |

---

## Model

- **Base model**: [`openai/whisper-small`](https://huggingface.co/openai/whisper-small)
- **Language**: Hindi (`hi`)
- **Task**: `transcribe`
- **Training config**: batch size 4, gradient checkpointing enabled, saved to Google Drive

---

## 📁 Project

**Notebook**: `Hindi ASR System — Fine-Tuning, Cleanup & Lattice Evaluation.ipynb`  
**Dataset**: Josh Talks — Hindi ASR Fine-tuning  
**Runtime**: Python 3 · Google Colab · GPU recommended
