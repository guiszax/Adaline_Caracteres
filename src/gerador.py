# ============================================================
# gerador.py
# Gera imagens sintéticas das 26 letras em Arial e Times New Roman
# ============================================================

import os
import math
from PIL import Image, ImageDraw, ImageFont

# ── Configurações ────────────────────────────────────────────
LETRAS      = [chr(c) for c in range(ord('A'), ord('Z') + 1)]
TAMANHO_IMG = 64          # pixels (largura e altura)
FONTE_SIZE  = 48          # tamanho da fonte
DIR_TREINO  = os.path.join(os.path.dirname(__file__), '..', 'dados', 'treino')
DIR_TESTE   = os.path.join(os.path.dirname(__file__), '..', 'dados', 'teste')

# Nomes das fontes — ajuste o caminho se necessário
FONTES = {
    "arial":           "arial.ttf",
    "times_new_roman": "times.ttf",
}

# Distorções aplicadas nas imagens de treino
DISTORCOES = [
    {"rotacao": 0,   "offset_x": 0,  "offset_y": 0},
    {"rotacao": 10,  "offset_x": 2,  "offset_y": 0},
    {"rotacao": -10, "offset_x": -2, "offset_y": 0},
    {"rotacao": 0,   "offset_x": 3,  "offset_y": 3},
    {"rotacao": 5,   "offset_x": -3, "offset_y": 2},
]


def _carregar_fonte(nome_arquivo: str, tamanho: int) -> ImageFont.FreeTypeFont:
    """Tenta carregar a fonte; usa fonte padrão se não encontrar."""
    try:
        return ImageFont.truetype(nome_arquivo, tamanho)
    except IOError:
        # Fallback para fonte padrão do Pillow (sem serifa)
        print(f"[AVISO] Fonte '{nome_arquivo}' não encontrada. Usando fonte padrão.")
        return ImageFont.load_default()


def _gerar_imagem(letra: str, fonte: ImageFont.FreeTypeFont,
                  rotacao: float = 0, offset_x: int = 0, offset_y: int = 0) -> Image.Image:
    """Cria uma imagem 64×64 com a letra centralizada e distorções opcionais."""
    img = Image.new("L", (TAMANHO_IMG, TAMANHO_IMG), color=255)   # fundo branco
    draw = ImageDraw.Draw(img)

    # Centraliza a letra
    bbox = draw.textbbox((0, 0), letra, font=fonte)
    larg = bbox[2] - bbox[0]
    alt  = bbox[3] - bbox[1]
    x = (TAMANHO_IMG - larg) // 2 + offset_x - bbox[0]
    y = (TAMANHO_IMG - alt)  // 2 + offset_y - bbox[1]

    draw.text((x, y), letra, fill=0, font=fonte)   # letra preta

    if rotacao != 0:
        img = img.rotate(rotacao, fillcolor=255, expand=False)

    return img


def gerar_dataset(verbose: bool = True) -> None:
    """Gera todas as imagens de treino e teste."""
    os.makedirs(DIR_TREINO, exist_ok=True)
    os.makedirs(DIR_TESTE,  exist_ok=True)

    total = 0
    for nome_fonte, arq_fonte in FONTES.items():
        fonte = _carregar_fonte(arq_fonte, FONTE_SIZE)

        for letra in LETRAS:
            # ── TREINO: letra com cada distorção ──
            for i, dist in enumerate(DISTORCOES):
                img = _gerar_imagem(
                    letra, fonte,
                    rotacao=dist["rotacao"],
                    offset_x=dist["offset_x"],
                    offset_y=dist["offset_y"],
                )
                nome = f"{letra}_{nome_fonte}_v{i}.png"
                img.save(os.path.join(DIR_TREINO, nome))
                total += 1

            # ── TESTE: letra sem distorção ──
            img_teste = _gerar_imagem(letra, fonte)
            nome_teste = f"{letra}_{nome_fonte}_teste.png"
            img_teste.save(os.path.join(DIR_TESTE, nome_teste))
            total += 1

    if verbose:
        print(f"[OK] Dataset gerado: {total} imagens")
        print(f"     Treino : {DIR_TREINO}")
        print(f"     Teste  : {DIR_TESTE}")


if __name__ == "__main__":
    gerar_dataset()
