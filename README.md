**Interactive WandB Report 👉** <https://wandb.ai/cs24m040-iit-madras/DA6401_Assignment_3/reports/Sai-Rajesh-s-Assignment-3--VmlldzoxMjg2NDU1Mg?accessToken=ymq984v9oy5y1qtj5v74v2pd7srt7uxiwc0wd04s5v10aiukcha2b9fkfxvok740>

# Transliteration Experiments on the Dakshina Dataset  
### Sequence-to-Sequence Baseline **vs.** Attention‑enhanced Model

---

## 1. Project Overview
This repository contains two independent but interoperable Python training scripts for character‑level English→Indian‑language transliteration:

| File | Model  | Attention | Decoder | Run Mode |
|------|--------|-----------|---------|----------|
| `transliteration_with_attention.py` | **Seq2Seq + Bahdanau Attention** | ✔️ | `AttentionDecoder` | default `run_transliteration_experiment` |
| `transliteration_without_attention.py` | **Plain Seq2Seq** | ❌ | `RNNDecoder` | default `run_transliteration_experiment` |

Both scripts are **self‑contained**: they download the [Dakshina v1.0](https://storage.googleapis.com/gresearch/dakshina/) dataset, build character vocabularies, set up dataloaders, start a training loop, evaluate on the Dev/Test splits and (optionally) launch a Weights & Biases hyper‑parameter sweep.

---

## 2. Quick Start

### 2.1 Prerequisites
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

<details>
<summary><code>requirements.txt</code></summary>

```
torch>=2.2
tqdm
pandas
wandb
```
</details>

### 2.2 Weights & Biases
Create a free [wandb](https://wandb.ai) account and **replace** the login key in each file:
```python
wandb.login(key='<YOUR_WANDB_API_KEY>')
```

---

## 3. Running a Single Experiment

> **GPU is auto‑detected** – if `torch.cuda.is_available()` returns `True`, the script uses CUDA.

```bash
# Attention model (recommended)
python transliteration_with_attention.py

# Baseline model
python transliteration_without_attention.py
```

By default each script trains a Telugu model with:

```python
{
  "rnn_type"       : "LSTM",
  "embedding_dim"  : 256,
  "hidden_dim"     : 512,
  "num_layers"     : 2,
  "bidirectional"  : True,
  "dropout"        : 0.3,
  "batch_size"     : 64,
  "epochs"         : 10,
  "learning_rate"  : 0.001,
  "teacher_forcing": 0.5,
  "optimizer"      : "Adam",
  "seed"           : 42
}
```

Model checkpoints are saved as  
`transliteration_model_attention_te_LSTM.pt` or `transliteration_model_te_LSTM.pt`.

---

## 4. Hyper‑parameter Sweeps

Each script exposes a **Bayesian** sweep configuration (`get_sweep_config()`) covering:

* Embedding & hidden sizes: 128 – 1024  
* RNN cell: RNN / GRU / LSTM  
* Directionality: uni vs. bi  
* Optimizer: Adam / NAdam  
* Training knobs: dropout, teacher‑forcing, LR, batch, epochs

Launch a 20‑run sweep:

```bash
python transliteration_with_attention.py   # or _without_attention.py
# The last line inside main():
# run_transliteration_experiment(use_wandb=True, run_sweep=True, sweep_count=20)
```

WandB automatically groups runs under **DA6401_Assignment_3** and tracks
* train / validation loss & accuracy  
* final test accuracy  
* sweep best config  

---

## 5. Code Highlights

### 5.1 `CharacterVocabulary`
* Adds `<pad> <bos> <eos> <unk>` tokens  
* Supports on‑disk caching (`cache/te_dakshina_vocab.pkl`)  
* Helper methods: `tokenize`, `detokenize`, `batch_detokenize`

### 5.2 Dataset & Dataloaders
* **Dakshina** TSV layout is *target tab source*.  
* Custom `create_batches` pads and packs sequences; lengths enable `pack_padded_sequence`.

### 5.3 Model Variants
|                | Encoder                                  | Decoder                               | Attention |
|----------------|------------------------------------------|---------------------------------------|-----------|
| **With Attn** | `RNNEncoder` (avg forward/backward state) | `AttentionDecoder` (emb + context)    | Bahdanau  |
| **Baseline**  | *same*                                   | `RNNDecoder` (emb only)               | —         |

Both share `Seq2SeqWithAttention` / `Seq2SeqModel` wrappers that implement:

* Teacher‑forcing loop (`ratio ∈ [0,1]`)  
* Greedy `generate()` for evaluation  
* Accuracy calculation & CSV dumps for correct/incorrect predictions  

---

## 6. Reproducibility

* `set_random_seeds(seed)` sets Python, CUDA & CuDNN deterministic flags  
* Each WandB run stores the seed value; sweeps explore multiple seeds (42‑46).

---

## 7. Outputs & Analysis

| Artifact | Description |
|----------|-------------|
| `model_*.pt` | Best checkpoint (highest Dev accuracy) |
| `correct_predictions_epoch_k.csv` | Per‑epoch correct pairs |
| `incorrect_predictions_epoch_k.csv` | Per‑epoch errors |
| `*_final.csv` | Final Test split break‑down |

Each CSV has `[Source, Target, Predicted]` columns for manual inspection.

---
