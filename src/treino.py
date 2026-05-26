# ============================================================
# treino.py
# Carrega dados, treina a rede Adaline e salva o modelo
# ============================================================

import os
import sys
import numpy as np

# Garante que src/ está no path
sys.path.insert(0, os.path.dirname(__file__))

from processamento import carregar_pasta
from adaline        import RedeAdaline

# ── Caminhos ─────────────────────────────────────────────────
BASE      = os.path.join(os.path.dirname(__file__), '..')
DIR_TREINO = os.path.join(BASE, 'dados', 'treino')
DIR_MODELO = os.path.join(BASE, 'modelos', 'adaline.pkl')


def treinar(
    taxa_aprendizado: float = 0.01,
    epocas:           int   = 200,
    tolerancia:       float = 0.01,
    resolucao:        int   = 28,
    callback=None,
) -> RedeAdaline:
    """
    Executa o pipeline completo de treinamento.
    Retorna a rede treinada.
    """
    print("\n══════════════════════════════════")
    print("  TREINAMENTO DA REDE ADALINE")
    print("══════════════════════════════════")
    print(f"  Taxa de aprendizado : {taxa_aprendizado}")
    print(f"  Épocas máximas      : {epocas}")
    print(f"  Tolerância (MSE)    : {tolerancia}")
    print(f"  Resolução           : {resolucao}×{resolucao}")
    print("══════════════════════════════════\n")

    # 1. Carrega imagens
    print("[1/3] Carregando imagens de treino...")
    X, y, _ = carregar_pasta(DIR_TREINO, resolucao=resolucao)
    print(f"      {len(X)} amostras | {X.shape[1]} atributos por amostra\n")

    # 2. Cria e treina a rede
    print("[2/3] Treinando 26 neurônios Adaline...")
    rede = RedeAdaline(n_entradas=X.shape[1],
                       taxa_aprendizado=taxa_aprendizado)

    def _progress(letra, idx, total):
        barra = "█" * idx + "░" * (total - idx)
        print(f"\r      [{barra}] {idx}/{total} — neurônio '{letra}'",
              end="", flush=True)
        if callback:
            callback(letra, idx, total)

    rede.treinar(X, y, epocas=epocas, tolerancia=tolerancia,
                 callback=_progress)
    print("\n")

    # 3. Salva modelo
    print("[3/3] Salvando modelo...")
    rede.salvar(DIR_MODELO)

    print("\n[CONCLUÍDO] Treinamento finalizado!\n")
    return rede


if __name__ == "__main__":
    treinar()
