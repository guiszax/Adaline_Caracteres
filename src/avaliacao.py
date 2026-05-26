# ============================================================
# avaliacao.py
# Avalia a rede no conjunto de teste e gera gráficos
# ============================================================

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")          # sem janela — salva direto em arquivo
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from processamento import carregar_pasta
from adaline        import RedeAdaline

BASE       = os.path.join(os.path.dirname(__file__), '..')
DIR_TESTE  = os.path.join(BASE, 'dados', 'teste')
DIR_MOD    = os.path.join(BASE, 'modelos', 'adaline.pkl')
DIR_RES    = os.path.join(BASE, 'resultados')


# ── Acurácia ─────────────────────────────────────────────────
def calcular_acuracia(rede: RedeAdaline, resolucao: int = 28) -> dict:
    X, y_real, _ = carregar_pasta(DIR_TESTE, resolucao=resolucao)
    y_pred        = rede.predizer_lote(X)

    acertos = sum(r == p for r, p in zip(y_real, y_pred))
    total   = len(y_real)
    acuracia_global = acertos / total * 100

    # Acurácia por letra
    por_letra = {}
    letras    = sorted(set(y_real))
    for letra in letras:
        idx     = [i for i, r in enumerate(y_real) if r == letra]
        certos  = sum(y_pred[i] == letra for i in idx)
        por_letra[letra] = certos / len(idx) * 100 if idx else 0.0

    return {
        "global":    acuracia_global,
        "por_letra": por_letra,
        "acertos":   acertos,
        "total":     total,
        "y_real":    y_real,
        "y_pred":    y_pred,
    }


# ── Gráfico 1: Curva de convergência (MSE) ───────────────────
def plot_convergencia(rede: RedeAdaline, salvar: bool = True) -> str:
    historico = rede.historico_erro_medio()
    if not historico:
        print("[AVISO] Histórico de erro vazio.")
        return ""

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(historico, color="steelblue", linewidth=1.8, label="MSE médio")
    ax.set_xlabel("Época", fontsize=11)
    ax.set_ylabel("Erro Quadrático Médio", fontsize=11)
    ax.set_title("Curva de Convergência — Rede Adaline", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    caminho = ""
    if salvar:
        os.makedirs(DIR_RES, exist_ok=True)
        caminho = os.path.join(DIR_RES, "convergencia.png")
        fig.savefig(caminho, dpi=150)
        print(f"[OK] Gráfico salvo: {caminho}")
    plt.close(fig)
    return caminho


# ── Gráfico 2: Acurácia por letra ────────────────────────────
def plot_acuracia_por_letra(resultado: dict, salvar: bool = True) -> str:
    por_letra = resultado["por_letra"]
    letras    = list(por_letra.keys())
    valores   = list(por_letra.values())

    fig, ax = plt.subplots(figsize=(12, 4))
    cores   = ["#4caf50" if v >= 80 else "#ff9800" if v >= 50 else "#f44336"
               for v in valores]
    ax.bar(letras, valores, color=cores, edgecolor="white", linewidth=0.5)
    ax.axhline(resultado["global"], color="steelblue",
               linestyle="--", linewidth=1.5,
               label=f"Acurácia global: {resultado['global']:.1f}%")
    ax.set_xlabel("Letra", fontsize=11)
    ax.set_ylabel("Acurácia (%)", fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_title("Acurácia por Letra — Conjunto de Teste", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    caminho = ""
    if salvar:
        os.makedirs(DIR_RES, exist_ok=True)
        caminho = os.path.join(DIR_RES, "acuracia_por_letra.png")
        fig.savefig(caminho, dpi=150)
        print(f"[OK] Gráfico salvo: {caminho}")
    plt.close(fig)
    return caminho


# ── Retorna figura Matplotlib (para embutir na interface) ─────
def figura_convergencia(rede: RedeAdaline):
    historico = rede.historico_erro_medio()
    fig, ax   = plt.subplots(figsize=(5, 3))
    if historico:
        ax.plot(historico, color="steelblue", linewidth=1.5)
    ax.set_xlabel("Época", fontsize=9)
    ax.set_ylabel("MSE", fontsize=9)
    ax.set_title("Convergência", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def figura_acuracia(resultado: dict):
    por_letra = resultado["por_letra"]
    letras    = list(por_letra.keys())
    valores   = list(por_letra.values())
    fig, ax   = plt.subplots(figsize=(10, 3))
    cores = ["#4caf50" if v >= 80 else "#ff9800" if v >= 50 else "#f44336"
             for v in valores]
    ax.bar(letras, valores, color=cores, edgecolor="white", linewidth=0.4)
    ax.axhline(resultado["global"], color="steelblue",
               linestyle="--", linewidth=1.2,
               label=f"Global: {resultado['global']:.1f}%")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Acurácia (%)", fontsize=9)
    ax.set_title("Acurácia por Letra", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


# ── CLI ───────────────────────────────────────────────────────
if __name__ == "__main__":
    rede      = RedeAdaline.carregar(DIR_MOD)
    resultado = calcular_acuracia(rede)

    print("\n══════════════════════════════════")
    print("  RESULTADOS DA AVALIAÇÃO")
    print("══════════════════════════════════")
    print(f"  Acertos : {resultado['acertos']} / {resultado['total']}")
    print(f"  Acurácia: {resultado['global']:.2f}%")
    print("══════════════════════════════════\n")
    for letra, acc in resultado["por_letra"].items():
        barra = "█" * int(acc / 5)
        print(f"  {letra}: {acc:6.1f}%  {barra}")

    plot_convergencia(rede)
    plot_acuracia_por_letra(resultado)
