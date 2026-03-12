# 📜 Arabic GPT — Classical Arabic Language Model

A GPT-style language model built from scratch in PyTorch, pre-trained on classical Arabic text, and fine-tuned for instruction following. Served via a Gradio chat interface.

---

## 🗂️ Project Structure

```
arabic-gpt/
├── star_classic_clean.ipynb   # Main notebook (all cells in order)
├── arabic_pretrain.txt        # Pre-training corpus (5k rows, classical Arabic)
├── aya_finetune.txt           # Fine-tuning data (10k Q&A pairs, Aya dataset)
├── arabic_gpt.pth             # Pre-trained model weights
└── arabic_gpt_finetuned.pth   # Fine-tuned model weights
```

---

## 🧱 Architecture — ArabicGPT

A decoder-only Transformer (same family as GPT-2), built from scratch using PyTorch.

```
Input tokens
     │
     ▼
Token Embedding (wte)  ←─── 64,000 vocab × 256 dim
     +
Position Embedding (wpe) ←── 128 positions × 256 dim
     │
     ▼
  Dropout
     │
     ▼

Final LayerNorm (ln_f)
     │
     ▼
LM Head (linear, no bias)   ← weight-tied with wte
     │
     ▼
Logits [batch, seq, 64000]
```

**Total parameters: ~21.16M**

### Hyperparameters

| Parameter | Value | Why |
|-----------|-------|-----|
| `vocab_size` | 64,000 | AraGPT2 tokenizer vocabulary |
| `context_length` | 128 | Max token sequence per sample |
| `emb_dim` | 256 | Embedding & hidden dimension |
| `n_heads` | 8 | Attention heads (256 ÷ 8 = 32 per head) |
| `n_layers` | 6 | Number of Transformer blocks |
| `drop_rate` | 0.1 | Dropout for regularization |
| `batch_size` | 16 | Samples per gradient step |
| `lr` | 4e-4 | Peak learning rate (AdamW) |
| `epochs` | 20 | Pre-training epochs |

---

## 🔬 Key Components Explained

### 1. Tokenizer — `aubmindlab/aragpt2-base`
We load only the **tokenizer** from AraGPT2 (weights are NOT used). It is a Byte-Pair Encoding (BPE) tokenizer trained specifically on Arabic text, giving us a 64,000-token vocabulary that handles Arabic morphology well.

```python
tokenizer = AutoTokenizer.from_pretrained("aubmindlab/aragpt2-base")
tokenizer.pad_token = tokenizer.eos_token  # reuse EOS as PAD
```

### 2. Causal Self-Attention
Multi-head attention where each token can only attend to itself and **previous** tokens (no future leaking). We use PyTorch's built-in `scaled_dot_product_attention(is_causal=True)` which handles the causal mask internally and is faster than a manual mask.

```
Q, K, V = linear(x).split(dim)          # project input
scores  = Q @ K.T / sqrt(head_dim)      # scaled dot product
scores  = mask(scores)                   # causal: future = -inf
weights = softmax(scores)
output  = weights @ V
```

### 3. MLP (Feed-Forward Block)
A two-layer network with a 4× expansion:
```
x → Linear(256 → 1024) → GELU → Linear(1024 → 256) → Dropout
```
GELU activation is smoother than ReLU and is standard in GPT-style models.

### 4. Pre-LayerNorm (Pre-LN)
Unlike the original Transformer ("Post-LN"), we normalize **before** each sub-layer, not after. Pre-LN makes training more stable and is used in GPT-2 and most modern LLMs.

```python
x = x + self.attn(self.ln_1(x))   # normalize → attend → add
x = x + self.mlp(self.ln_2(x))    # normalize → MLP → add
```

### 5. Weight Tying
The token embedding matrix (`wte`) and the output projection (`lm_head`) **share the same weights**. This reduces parameters and improves performance — the model learns that similar tokens should have similar input embeddings AND similar output probabilities.

```python
self.transformer.wte.weight = self.lm_head.weight
```

---

## 📊 Training Pipeline

### Phase 1 — Pre-training (Causal Language Modeling)

**Data:** `AbderrahmanSkiredj1/MLM_classical_arabic_openiti` — 50,000 rows of classical Arabic text from OpenITI (a large corpus of digitized Islamic and Arabic literature).

**Method:** Sliding window over the token stream with stride=128. For each window of 128 tokens, the input is tokens `[0:128]` and the target is tokens `[1:129]` — the model learns to predict the next token at every position.

**Loss:** Cross-Entropy between predicted logits and true next tokens.

**Split:** 80% train / 20% validation.

**Optimizer:** AdamW with weight decay 0.4 and cosine annealing LR schedule.

```
Epoch 1/3 | Train: 7.4946 | Val: 6.8291
Epoch 2/3 | Train: 6.6403 | Val: 6.4824
Epoch 3/3 | Train: 6.3438 | Val: 6.3491
```
*(Loss measured in nats; lower is better. More epochs needed for strong generation.)*

### Phase 2 — Instruction Fine-tuning

**Data:** `2A2I/Arabic_Aya` (CohereForAI Aya dataset) — 10,000 Arabic Q&A pairs.

**Template:** Each sample is formatted as:
```
### السؤال: {question}
### الإجابة: {answer} <EOS>
```

This teaches the model to associate the pattern `### السؤال:` with producing an answer after `### الإجابة:`.

**Optimizer:** AdamW with a lower LR of `5e-5` (fine-tuning requires smaller updates to not overwrite pre-trained knowledge).

**Loss trend:** 13.4 → 0.7 over 1 epoch (625 batches), showing rapid adaptation.

---

## ⚙️ GPTCoach — Training Wrapper

A helper class that bundles the model with its optimizer, scheduler, and generation logic.

| Method | What it does |
|--------|-------------|
| `train_epoch(loader)` | One full pass over training data, gradient clipping at 1.0 |
| `validate(loader)` | Evaluation pass with `torch.no_grad()` |
| `generate(prompt, tokens_to_add, temp, top_k)` | Top-k + temperature sampling |

### Text Generation — Top-k Sampling
At each step:
1. Run the model on the current sequence → get logits for next token
2. Divide logits by `temperature` (lower = more confident, higher = more creative)
3. Keep only the top-k logits, set the rest to `-inf`
4. Sample from the resulting softmax distribution
5. Stop early if the EOS token is generated

---

## 🖥️ Chat Interface — Gradio

A `gr.ChatInterface` wraps the fine-tuned model. The user's message is injected into the instruction template before generation, and only the answer portion (after `### الإجابة:`) is returned.

**Controls:**
- **Temperature (الإبداع):** 0.1–1.2 — controls randomness of generation
- **Max Tokens (طول النص):** 10–300 — max new tokens to generate

---

## 🛠️ Libraries & Tools

| Library | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.x | Model definition, training, GPU acceleration |
| `transformers` | 4.x | AraGPT2 tokenizer only |
| `datasets` | latest | Loading Aya dataset from HuggingFace Hub |
| `pandas` | latest | Loading and formatting the parquet dataset |
| `gradio` | latest | Web-based chat interface |
| `matplotlib` | latest | Loss curve visualization |

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install torch transformers datasets pandas gradio matplotlib
```

### 2. Run in order (Google Colab recommended for GPU)
| Cell | Description |
|------|-------------|
| Cell 1 | Imports, device setup, tokenizer, config |
| Cell 2 | Load & clean dataset, create DataLoaders |
| Cell 3 | Define model architecture |
| Cell 4 | Define GPTCoach (trainer + generator) |
| Cell 5 | Pre-training loop + loss plot |
| Cell 6 | Save weights to disk |
| Cell 7 | Load weights from disk |
| Cell 8 | Load Aya dataset, create fine-tuning DataLoader |
| Cell 9 | Fine-tuning loop |
| Cell 10 | Launch Gradio chat interface |

### 3. Tips
- **Use a GPU** (T4 on Colab is enough). CPU training is ~50× slower.
- Run **at least 10–20 pre-training epochs** before fine-tuning for coherent output.
- If the model generates padding tokens (`---`), it needs more pre-training before fine-tuning.
- Set `HF_TOKEN` in Colab secrets to avoid HuggingFace rate limits.

---

## 📈 Results & Known Limitations

- The model is small (~21M params) compared to production Arabic LLMs (billions of params)
- Pre-training on only 50,000 rows is limited — more data = better fluency
- Generation quality improves significantly with more epochs and data
- The model generates **classical/formal Arabic** (فصحى) — it may struggle with dialects

---

## 📚 References & Datasets

- **OpenITI Arabic Corpus:** [AbderrahmanSkiredj1/MLM_classical_arabic_openiti](https://huggingface.co/datasets/AbderrahmanSkiredj1/MLM_classical_arabic_openiti)
- **Aya Dataset (Arabic):** [2A2I/Arabic_Aya](https://huggingface.co/datasets/2A2I/Arabic_Aya)
- **AraGPT2 Tokenizer:** [aubmindlab/aragpt2-base](https://huggingface.co/aubmindlab/aragpt2-base)
- **GPT-2 Paper:** Radford et al., 2019 — *Language Models are Unsupervised Multitask Learners*
- **Attention is All You Need:** Vaswani et al., 2017
