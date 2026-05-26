# ============================================================
# processamento.py
# Pré-processa imagens: cinza → binarização → resize → vetor
# ============================================================

import os
import cv2
import numpy as np
from PIL import Image

RESOLUCAO = 28   # padrão; pode ser alterado pelo usuário


def imagem_para_vetor(caminho: str, resolucao: int = RESOLUCAO) -> np.ndarray:
    """
    Lê uma imagem, binariza e retorna vetor 1-D com valores ±1.
    +1 = pixel pertence à letra
    -1 = pixel pertence ao fundo
    """
    img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {caminho}")

    # Binarização com limiar 127
    _, binaria = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Redimensiona para resolucao × resolucao
    redim = cv2.resize(binaria, (resolucao, resolucao),
                       interpolation=cv2.INTER_AREA)

    # Normaliza: fundo(255) → -1 | letra(0) → +1
    vetor = np.where(redim < 127, 1.0, -1.0).flatten()
    return vetor


def pil_para_vetor(img_pil: Image.Image, resolucao: int = RESOLUCAO) -> np.ndarray:
    """Converte imagem PIL (vinda da interface) para vetor ±1."""
    cinza   = img_pil.convert("L")
    redim   = cinza.resize((resolucao, resolucao), Image.LANCZOS)
    arr     = np.array(redim)
    _, bin_ = cv2.threshold(arr, 127, 255, cv2.THRESH_BINARY)
    vetor   = np.where(bin_ < 127, 1.0, -1.0).flatten()
    return vetor


def carregar_pasta(pasta: str, resolucao: int = RESOLUCAO):
    """
    Carrega todas as imagens de uma pasta.
    Retorna:
        X : ndarray (n_amostras, resolucao²)
        y : list[str]  rótulos (letra, ex: 'A')
        arquivos : list[str]
    """
    X, y, arquivos = [], [], []

    for nome in sorted(os.listdir(pasta)):
        if not nome.lower().endswith(".png"):
            continue
        caminho = os.path.join(pasta, nome)
        letra   = nome[0].upper()          # primeiro caractere = rótulo
        vetor   = imagem_para_vetor(caminho, resolucao)
        X.append(vetor)
        y.append(letra)
        arquivos.append(nome)

    return np.array(X), y, arquivos


if __name__ == "__main__":
    # Teste rápido
    base = os.path.join(os.path.dirname(__file__), '..', 'dados', 'treino')
    X, y, _ = carregar_pasta(base)
    print(f"[OK] {len(X)} amostras carregadas | shape: {X.shape}")
    print(f"     Letras únicas: {sorted(set(y))}")
