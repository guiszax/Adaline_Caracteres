# ============================================================
# adaline.py
# Implementação da rede Adaline com 26 neurônios (One-vs-All)
# ============================================================

import numpy as np
import pickle
import os

LETRAS = [chr(c) for c in range(ord('A'), ord('Z') + 1)]


class NeuronioAdaline:
    """
    Um único neurônio Adaline — classificador binário bipolar.
    Aprende a responder +1 para UMA letra e -1 para todas as outras.
    """

    def __init__(self, n_entradas: int, taxa_aprendizado: float = 0.01):
        # Pesos inicializados com valores aleatórios pequenos
        self.w  = np.random.uniform(-0.5, 0.5, n_entradas)
        self.b  = np.random.uniform(-0.5, 0.5)        # bias
        self.lr = taxa_aprendizado
        self.historico_erro = []                       # MSE por época

    # ── Saída linear (antes da função de ativação) ───────────
    def saida_linear(self, x: np.ndarray) -> float:
        return float(np.dot(self.w, x) + self.b)

    # ── Função degrau bipolar ─────────────────────────────────
    @staticmethod
    def degrau(y: float) -> int:
        return 1 if y >= 0 else -1

    # ── Predição final (+1 ou -1) ─────────────────────────────
    def predizer(self, x: np.ndarray) -> int:
        return self.degrau(self.saida_linear(x))

    # ── Treinamento (Regra Delta) ─────────────────────────────
    def treinar(self, X: np.ndarray, targets: np.ndarray,
                epocas: int = 200, tolerancia: float = 0.01) -> None:
        """
        X       : matriz (n_amostras, n_entradas)
        targets : vetor  (n_amostras,) com valores +1 ou -1
        """
        self.historico_erro = []

        for epoca in range(epocas):
            erros_quadraticos = []

            for x, t in zip(X, targets):
                y      = self.saida_linear(x)      # saída linear
                erro   = t - y                     # erro antes do degrau
                # Regra Delta: atualiza pesos e bias
                self.w += self.lr * erro * x
                self.b += self.lr * erro
                erros_quadraticos.append(erro ** 2)

            mse = float(np.mean(erros_quadraticos))
            self.historico_erro.append(mse)

            if mse <= tolerancia:
                break


class RedeAdaline:
    """
    Rede com 26 neurônios Adaline — um por letra (One-vs-All).
    """

    def __init__(self, n_entradas: int, taxa_aprendizado: float = 0.01):
        self.letras    = LETRAS
        self.neuronios = {
            letra: NeuronioAdaline(n_entradas, taxa_aprendizado)
            for letra in self.letras
        }
        self.n_entradas        = n_entradas
        self.taxa_aprendizado  = taxa_aprendizado
        self.historico_acuracia = []

    # ── Treina todos os 26 neurônios ──────────────────────────
    def treinar(self, X: np.ndarray, y_rotulos: list,
                epocas: int = 200, tolerancia: float = 0.01,
                callback=None) -> None:
        """
        X        : (n_amostras, n_entradas)
        y_rotulos: lista de letras, ex: ['A','A','B','C',...]
        callback : função opcional chamada a cada letra treinada
                   callback(letra, indice, total)
        """
        for i, letra in enumerate(self.letras):
            # Target: +1 se for essa letra, -1 se for qualquer outra
            targets = np.array([1.0 if r == letra else -1.0
                                 for r in y_rotulos])
            self.neuronios[letra].treinar(X, targets, epocas, tolerancia)

            if callback:
                callback(letra, i + 1, len(self.letras))

    # ── Classifica uma amostra ────────────────────────────────
    def predizer(self, x: np.ndarray) -> str:
        """Retorna a letra cuja saída linear for a maior."""
        saidas = {
            letra: self.neuronios[letra].saida_linear(x)
            for letra in self.letras
        }
        return max(saidas, key=saidas.get)

    def predizer_lote(self, X: np.ndarray) -> list:
        return [self.predizer(x) for x in X]

    # ── Histórico de erro médio (todos os neurônios) ──────────
    def historico_erro_medio(self) -> list:
        """MSE médio de todos os neurônios por época."""
        historicos = [self.neuronios[l].historico_erro for l in self.letras]
        if not historicos or not historicos[0]:
            return []
        min_len = min(len(h) for h in historicos)
        return [float(np.mean([h[e] for h in historicos]))
                for e in range(min_len)]

    # ── Salva e carrega o modelo ──────────────────────────────
    def salvar(self, caminho: str) -> None:
        os.makedirs(os.path.dirname(caminho) or ".", exist_ok=True)
        with open(caminho, "wb") as f:
            pickle.dump(self, f)
        print(f"[OK] Modelo salvo em: {caminho}")

    @staticmethod
    def carregar(caminho: str) -> "RedeAdaline":
        with open(caminho, "rb") as f:
            rede = pickle.load(f)
        print(f"[OK] Modelo carregado de: {caminho}")
        return rede


if __name__ == "__main__":
    # Teste básico com dados aleatórios
    np.random.seed(42)
    X_fake = np.random.uniform(-1, 1, (52, 784))
    y_fake = LETRAS * 2
    rede   = RedeAdaline(n_entradas=784, taxa_aprendizado=0.01)
    rede.treinar(X_fake, y_fake, epocas=10)
    print("[OK] Teste básico do Adaline concluído.")
    print(f"     Predição de amostra 0: {rede.predizer(X_fake[0])}")
