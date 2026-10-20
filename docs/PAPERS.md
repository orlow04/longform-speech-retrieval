# Papers and target numbers

The reproduction targets, so they live here rather than in a PDF. Numbers are
what the papers report; nothing below has been independently reproduced unless
[LANDSCAPE.md](LANDSCAPE.md) says so.

---

## CLSR — *End-to-end Contrastive Language-Speech Pretraining Model for Long-form Spoken Question Answering*

AAAI-26. Hu, Li, Qi, Guoming, Wang (Wuhan University + Xiaomi).
Code: `github.com/193746/CLSR`.

**Claim.** The dual-encoder speech-text contrastive design is wrong: aligning
acoustic embeddings to text embeddings directly is too hard without massive
pretraining. Convert acoustic features into *text-like* embeddings first, then
do the contrastive alignment inside the text modality, borrowing a strong
frozen text retriever's generalisation.

**Architecture** (Paraformer 220M + frozen BGE-base 109M):

1. SAN-M speech encoder → `H^s`
2. **CIF** (continuous integrate-and-fire) resamples from time steps to token
   count → `E^a`, aligned with the token probability distribution. Weights
   `α = W·conv(H^s) + b` accumulate until they cross threshold β; on crossing,
   the overflow carries into the next accumulation.
3. Speech decoder + linear → token probability distribution `D`
4. **VQ adaptor**: `q_i = argmax_j D_ij`; temperature-softmax
   `p̃ = softmax(D/γ)` with γ = 0.1; straight-through estimator
   `q^st = q + p̃ − sg(p̃)`; then matmul the resulting one-hot matrix with the
   text encoder's embedding weights `W_te` → text-like embeddings
5. Frozen text encoder consumes real or text-like embeddings; CLS-token
   sentence representation; cosine similarity
6. **Sampler** (two-round training): round 1 produces an ASR transcript
   greedily; round 2 replaces a λ fraction of the *erroneous* tokens' acoustic
   embeddings with correct text embeddings — taken from the weights of the
   **linear output layer**, not the decoder's embedding layer, because the
   latter is not trained
7. Loss: `L = (1−α−β)·L_ASR + α·L_MAE + β·L_NLL` with α = β = 1/3, where
   `L_ASR = γ·L_CE + L_MWER`. Adam, lr 5e-5. MAE drives CIF convergence.

### Targets, each tagged with its split

**Table 2 — Spoken-SQuAD\*** (filtered one-to-one, 29,227 train / 3,884 val),
Q→C R@1 / R@5 / R@10:

| System | R@1 | R@5 | R@10 | WER |
|---|---|---|---|---|
| CLSR | 70.03 | 86.90 | 90.68 | 15.14 |
| Whisper + BGE (cascade) | 69.93 | 86.61 | 90.53 | 19.39 |
| BGE (clean text, reference ceiling) | 67.12 | 85.20 | 89.44 | 0 |
| CLAP | 2.93 | 9.92 | 14.84 | — |

**Table 2 — other datasets** (Q→C R@1): LibriSQA 85.04 (WER 4.09);
SLUE-SQA-5 30.65 (WER 16.69); DRCD\* 76.21 (WER 5.56).

**Tables 3–4 — unfiltered Spoken-SQuAD** (37,107 / 5,351): CLSR 49.82,
**ParaBGE 17.79**. ParaBGE is Paraformer's encoder + BGE's encoder as a dual
encoder at matched parameter count — the paper's central evidence, and the
direct analogue of CLASP. The load-bearing ablation is **no-VQ**: without the
quantizer, R@10 falls to roughly the R@1 of the full model (15.29 vs 49.82 R@1).

**Table 5 — long-form SQA**: 500 randomly selected SLUE-SQA-5 test instances
with their context audio replaced by full ~30-minute Spoken Wikipedia
documents; segment into 40-second intervals, select the most query-relevant,
feed Qwen-Audio.

| | EM | F1 | Cost (s) | Speedup |
|---|---|---|---|---|
| Qwen-Audio alone | 18.00 | 23.55 | 7,935 | 1.00× |
| + CLSR | 27.60 | 35.10 | 783 | 10.13× |

### Traps

- **Table 2 and Tables 3–4 are different datasets.** 70.03 is Spoken-SQuAD\*
  (filtered); 49.82 is unfiltered Spoken-SQuAD. Same model, same metric, ~20
  points apart. Every number reported here must name its split.
- **Qwen-Audio and Qwen2-Audio truncate at 30 s** (3,000 mel frames,
  Whisper-large-v2 initialisation). So Table 5's baseline, fed a 30-minute
  document, sees only its first 30 seconds — a truncation baseline, not a
  long-context one — and CLSR's 40-second selected segment is itself cut to
  30 s. The 10.13× speedup (15.9 s → 1.57 s per instance) is hard to explain if
  both arms deliver ≤ 30 s of audio. Any long-audio arm must report the seconds
  actually delivered to the model.
- The release does not match the paper; see [LANDSCAPE.md](LANDSCAPE.md).

---

## ParaSpeechRAG — *A Benchmark for Robust Speech Retrieval under Paralinguistic and Acoustic Perturbations*

Our own submission. The artifact repo owns this one; summarised here only
because its perturbation axes and its CLASP numbers are reference points.

A counterfactual robustness benchmark: apply perturbation `φ` to the query
audio only, leaving the corpus untouched and lexical content fixed, and measure
how far retrieval moves. Five axes — speaker identity (Seed-VC and GenVC, six
target voices), emotion and prosody (Seed-VC v2, RAVDESS, 7 conditions), noise
(white, ESC-50 ambient, reverb at 20/10/5 dB), speaking rate (librosa
`time_stretch`, 0.5×–2.0×), codec (MP3 and Opus, 8–128 kbps).

Headline: CLASP-mean leads in clean conditions (R@1 0.954, MRR 0.971) but
degrades sharply under perturbation — strong-happy emotion 0.651, ambient noise
at 5 dB 0.301, 2.0× rate 0.630 — while the ASR cascade stays near 0.694
throughout, because the text bottleneck discards paralinguistic information.
Codec compression is nearly harmless; reverberation saturates early rather than
worsening with severity.

Caveat carried over from [LANDSCAPE.md](LANDSCAPE.md): the underlying task is
self-retrieval, not QA retrieval, so the 0.954 is not comparable to CLSR's
70.03.

---

## Datasets

| Dataset | Where |
|---|---|
| Spoken-SQuAD | `arXiv:1804.00320`; TTS-read SQuAD articles, text questions |
| LibriSQA | `ZihanZhaoSJTU/LibriSQA`, part 1 (open questions) + LibriSpeech audio |
| SLUE-SQA-5 | SLUE phase-2; both questions and contexts are real recordings |
| Spoken Wikipedia Corpus | CLARIN, `hdl.handle.net/11022/0000-0007-C641-0`; English 287 h |
| ESC-50 | environmental noise for the ambient axis |
| DRCD | Chinese; spoken dev set from ODSQA, train set TTS-synthesised |
