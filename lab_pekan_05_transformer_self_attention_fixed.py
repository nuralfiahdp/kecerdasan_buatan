"""
========================================================================================
LABORATORIUM KECERDASAN BUATAN (MK55) — PEKAN KE-5
UNIVERSITAS ISLAM NEGERI (UIN) ALAUDDIN MAKASSAR
FAKULTAS SAINS DAN TEKNOLOGI • PROGRAM STUDI SISTEM INFORMASI
========================================================================================
Topik         : Arsitektur Transformer & Mekanisme Scaled Dot-Product Self-Attention
SubCPMK       : SubCPMK 04 (CPMK-2 / CPL02)
Tugas Terkait : Tugas Analisis 1 — Diagram Alur Transformer & Kalkulasi Matriks Q, K, V
Bobot Nilai   : 5% dari Total Nilai Semester
Dosen Pengampu: Adhy Rizaldy, M.Kom. & Muhammad Rezky, S.Pd., M.Kom.
========================================================================================

PETUNJUK PRAKTIKUM:
Skrip ini mengimplementasikan mekanisme fundamental Transformer dari prinsip matematika
pertama (first principles) menggunakan pustaka murni NumPy dan visualisasi Matplotlib:
1. Positional Encoding Sinusoidal (Vaswani et al., 2017)
2. Kalkulasi Scaled Dot-Product Attention: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
3. Causal Masking (Look-Ahead Mask) untuk Lapisan Decoder
4. Mekanisme Multi-Head Self-Attention (Parallel Projections & Concatenation)
5. Studi Kasus Resolusi Ambiguitas Konteks ("animal" vs "street" pada kata ganti "it")
6. Visualisasi Matriks Bobot Perhatian (Attention Heatmap) yang disimpan otomatis ke disk.
========================================================================================
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

np.random.seed(42)


# ==================================================
# 1. FUNGSI DASAR
# ==================================================
def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def get_positional_encoding(seq_len, d_model):
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, np.newaxis]

    div_term = np.exp(
        np.arange(0, d_model, 2)
        * -(np.log(10000.0) / d_model)
    )

    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    return pe


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = K.shape[-1]

    scores = Q @ K.T
    scaled_scores = scores / np.sqrt(d_k)

    if mask is not None:
        scaled_scores = np.where(mask == 1, -1e9, scaled_scores)

    weights = softmax(scaled_scores, axis=-1)
    output = weights @ V

    return output, weights


def create_causal_mask(seq_len):
    return np.triu(np.ones((seq_len, seq_len)), k=1)


# ==================================================
# 2. NOMOR 1: PERHITUNGAN SCALED DOT-PRODUCT ATTENTION
# ==================================================
def run_manual_attention_example():
    print("\n" + "=" * 60)
    print("NOMOR 1: SCALED DOT-PRODUCT ATTENTION")
    print("=" * 60)

    Q = np.array([
        [1, 0],
        [0, 1],
        [1, 1]
    ], dtype=float)

    K = np.array([
        [1, 0],
        [0, 1],
        [1, 1]
    ], dtype=float)

    V = np.array([
        [1, 0],
        [0, 1],
        [1, 1]
    ], dtype=float)

    # Langkah 1: Q dikali transpose K
    scores = Q @ K.T

    print("\n1. Hasil QK^T:")
    print(scores)

    # Langkah 2: scaling
    scaled_scores = scores / np.sqrt(K.shape[1])

    print("\n2. Hasil QK^T / akar(d_k):")
    print(np.round(scaled_scores, 3))

    # Langkah 3: Softmax
    weights = softmax(scaled_scores, axis=1)

    print("\n3. Bobot perhatian setelah Softmax:")
    print(np.round(weights, 3))

    # Langkah 4: Output Attention
    output = weights @ V

    print("\n4. Output Attention:")
    print(np.round(output, 3))


# ==================================================
# 3. MULTI-HEAD ATTENTION
# ==================================================
class SimpleMultiHeadAttention:
    def __init__(self, d_model=16, num_heads=2):
        if d_model % num_heads != 0:
            raise ValueError(
                "d_model harus habis dibagi num_heads"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = np.random.randn(d_model, d_model) * 0.05
        self.W_k = np.random.randn(d_model, d_model) * 0.05
        self.W_v = np.random.randn(d_model, d_model) * 0.05
        self.W_o = np.random.randn(d_model, d_model) * 0.05

    def split_heads(self, x):
        seq_len = x.shape[0]
        x = x.reshape(seq_len, self.num_heads, self.d_k)
        return np.transpose(x, (1, 0, 2))

    def forward(self, x):
        Q = self.split_heads(x @ self.W_q)
        K = self.split_heads(x @ self.W_k)
        V = self.split_heads(x @ self.W_v)

        outputs = []
        all_weights = []

        for head in range(self.num_heads):
            output, weights = scaled_dot_product_attention(
                Q[head], K[head], V[head]
            )

            outputs.append(output)
            all_weights.append(weights)

        combined = np.concatenate(outputs, axis=-1)
        final_output = combined @ self.W_o

        return final_output, np.array(all_weights)


# ==================================================
# 4. SELF-ATTENTION KALIMAT BAHASA INDONESIA
# ==================================================
def run_indonesian_case():
    print("\n" + "=" * 60)
    print("STUDI KASUS SELF-ATTENTION")
    print("=" * 60)

    tokens = [
        "Mahasiswa",
        "sedang",
        "belajar",
        "kecerdasan",
        "buatan",
        "di",
        "kampus"
    ]

    print('Kalimat: "' + " ".join(tokens) + '"')
    print("Jumlah token:", len(tokens))

    seq_len = len(tokens)
    d_model = 16

    # Embedding sintetis untuk demonstrasi
    embeddings = np.random.randn(seq_len, d_model) * 0.1

    W_q = np.random.randn(d_model, d_model) * 0.2
    W_k = np.random.randn(d_model, d_model) * 0.2
    W_v = np.random.randn(d_model, d_model) * 0.2

    Q = embeddings @ W_q
    K = embeddings @ W_k
    V = embeddings @ W_v

    # Penyesuaian sintetis untuk demonstrasi
    idx_belajar = tokens.index("belajar")
    idx_kecerdasan = tokens.index("kecerdasan")
    idx_buatan = tokens.index("buatan")
    idx_kampus = tokens.index("kampus")

    K[idx_kecerdasan] = (
        Q[idx_belajar] * 1.8
        + np.random.randn(d_model) * 0.05
    )

    K[idx_buatan] = (
        Q[idx_belajar] * 1.2
        + np.random.randn(d_model) * 0.05
    )

    K[idx_kampus] = (
        Q[idx_belajar] * 0.3
        + np.random.randn(d_model) * 0.05
    )

    _, attention_weights = scaled_dot_product_attention(
        Q, K, V
    )

    row = attention_weights[idx_belajar]

    print("\nBobot perhatian untuk token 'belajar':")

    for idx in np.argsort(row)[::-1]:
        print(
            f"- {tokens[idx]:<12}: "
            f"{row[idx]:.4f} ({row[idx] * 100:.2f}%)"
        )

    highest = int(np.argmax(row))

    print(
        f"\nBobot tertinggi diberikan kepada "
        f"'{tokens[highest]}' sebesar {row[highest] * 100:.2f}%."
    )

    print(
        "Catatan: hasil menggunakan embedding sintetis, "
        "bukan model bahasa yang telah dilatih."
    )

    return tokens, attention_weights


# ==================================================
# 5. MEMBUAT HEATMAP
# ==================================================
def save_attention_heatmap(tokens, attention_weights):
    output_path = (
        Path(__file__).resolve().parent
        / "attention_heatmap_pekan_05.png"
    )

    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)

    image = ax.imshow(attention_weights, cmap="viridis")
    fig.colorbar(image, ax=ax, label="Bobot Perhatian")

    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))

    ax.set_xticklabels(tokens, rotation=45, ha="left")
    ax.set_yticklabels(tokens)

    ax.set_xlabel("Key Tokens (K)")
    ax.set_ylabel("Query Tokens (Q)")
    ax.set_title("Heatmap Self-Attention Bahasa Indonesia")

    for i in range(len(tokens)):
        for j in range(len(tokens)):
            value = attention_weights[i, j]

            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value < 0.4 else "black",
                fontsize=8
            )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print("\nHeatmap tersimpan di:")
    print(output_path)


# ==================================================
# 6. DEMONSTRASI CAUSAL MASK
# ==================================================
def run_causal_mask_demo():
    print("\n" + "=" * 60)
    print("DEMONSTRASI CAUSAL MASK")
    print("=" * 60)

    tokens = [
        "Saya",
        "sedang",
        "belajar",
        "Transformer",
        "AI"
    ]

    n = len(tokens)
    mask = create_causal_mask(n)

    print("\nMatriks Causal Mask:")
    print(mask)

    scores = np.random.randn(n, n)
    masked_scores = np.where(mask == 1, -1e9, scores)
    weights = softmax(masked_scores, axis=1)

    print("\nBobot perhatian setelah Causal Mask:")

    for i, token in enumerate(tokens):
        values = " ".join(
            f"{value:.4f}" for value in weights[i]
        )
        print(f"Query '{token}': [ {values} ]")

    print("\nBobot pada posisi token masa depan menjadi 0.")


# ==================================================
# 7. PROGRAM UTAMA
# ==================================================
def main():
    print("Praktikum Pekan 5: Transformer & Self-Attention")

    print("\nContoh Positional Encoding:")
    print(np.round(get_positional_encoding(6, 8), 3))

    sample = np.random.randn(4, 16)

    mha = SimpleMultiHeadAttention(
        d_model=16,
        num_heads=2
    )

    mha_output, mha_weights = mha.forward(sample)

    print("\nPengujian Multi-Head Attention:")
    print("Bentuk input :", sample.shape)
    print("Bentuk output:", mha_output.shape)
    print("Bentuk bobot :", mha_weights.shape)

    # Nomor 1: perhitungan khusus
    run_manual_attention_example()

    # Self-Attention dan heatmap
    tokens, weights = run_indonesian_case()
    save_attention_heatmap(tokens, weights)

    # Causal Mask
    run_causal_mask_demo()

    print("\nPraktikum selesai dijalankan tanpa error.")


if __name__ == "__main__":
    main()