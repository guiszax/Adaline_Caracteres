# ============================================================
# interface.py
# Interface gráfica com CustomTkinter
# ============================================================

import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Garante que src/ está no path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from src.gerador       import gerar_dataset
from src.processamento import pil_para_vetor, carregar_pasta
from src.adaline       import RedeAdaline
from src.treino        import treinar as executar_treino
from src.avaliacao     import (calcular_acuracia, figura_convergencia,
                               figura_acuracia)

# ── Configuração visual ──────────────────────────────────────
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

MODELO_PATH = os.path.join(BASE_DIR, "modelos", "adaline.pkl")
CINZA_ESCURO = "#2b2b2b"
CINZA_MEDIO  = "#444444"
CINZA_CLARO  = "#f0f0f0"


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Reconhecimento de Caracteres — Adaline")
        self.geometry("1000x700")
        self.resizable(True, True)
        self.rede: RedeAdaline | None = None
        self._construir_ui()
        self._tentar_carregar_modelo()

    # ════════════════════════════════════════════════════════
    #  CONSTRUÇÃO DA UI
    # ════════════════════════════════════════════════════════
    def _construir_ui(self):
        # Barra lateral esquerda
        self.sidebar = ctk.CTkFrame(self, width=240, corner_radius=0)
        self.sidebar.pack(side="left", fill="y", padx=0, pady=0)
        self.sidebar.pack_propagate(False)

        ctk.CTkLabel(self.sidebar, text="🔤 Adaline",
                     font=ctk.CTkFont(size=20, weight="bold")).pack(pady=(20, 4))
        ctk.CTkLabel(self.sidebar, text="Reconhecimento de Caracteres",
                     font=ctk.CTkFont(size=11), text_color="gray").pack(pady=(0, 20))

        # ── Parâmetros ──
        ctk.CTkLabel(self.sidebar, text="Parâmetros",
                     font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=16, pady=(8, 2))

        ctk.CTkLabel(self.sidebar, text="Taxa de aprendizado (α)").pack(anchor="w", padx=16)
        self.var_alpha = ctk.StringVar(value="0.01")
        ctk.CTkOptionMenu(self.sidebar, variable=self.var_alpha,
                          values=["0.1", "0.01", "0.001"]).pack(fill="x", padx=16, pady=(0, 8))

        ctk.CTkLabel(self.sidebar, text="Resolução da imagem").pack(anchor="w", padx=16)
        self.var_resolucao = ctk.StringVar(value="28×28")
        ctk.CTkOptionMenu(self.sidebar, variable=self.var_resolucao,
                          values=["10×10", "20×20", "28×28"]).pack(fill="x", padx=16, pady=(0, 8))

        ctk.CTkLabel(self.sidebar, text="Épocas máximas").pack(anchor="w", padx=16)
        self.var_epocas = ctk.CTkEntry(self.sidebar)
        self.var_epocas.insert(0, "200")
        self.var_epocas.pack(fill="x", padx=16, pady=(0, 16))

        # ── Botões de ação ──
        ctk.CTkButton(self.sidebar, text="1️⃣  Gerar imagens",
                      command=self._gerar_imagens).pack(fill="x", padx=16, pady=4)
        ctk.CTkButton(self.sidebar, text="2️⃣  Treinar rede",
                      command=self._treinar).pack(fill="x", padx=16, pady=4)
        ctk.CTkButton(self.sidebar, text="3️⃣  Avaliar rede",
                      command=self._avaliar).pack(fill="x", padx=16, pady=4)

        ctk.CTkLabel(self.sidebar, text="").pack(expand=True)
        ctk.CTkLabel(self.sidebar, text="IFB — Adaline",
                     font=ctk.CTkFont(size=10), text_color="gray").pack(pady=8)

        # ── Área principal (tabs) ──
        self.tabview = ctk.CTkTabview(self, corner_radius=8)
        self.tabview.pack(side="right", fill="both", expand=True, padx=12, pady=12)

        self.tab_teste  = self.tabview.add("🔍 Testar letra")
        self.tab_result = self.tabview.add("📊 Resultados")
        self.tab_log    = self.tabview.add("📋 Log")

        self._construir_tab_teste()
        self._construir_tab_resultados()
        self._construir_tab_log()

    # ── Tab: Testar letra ─────────────────────────────────────
    def _construir_tab_teste(self):
        frame = self.tab_teste

        ctk.CTkLabel(frame, text="Carregar imagem de uma letra e classificar",
                     font=ctk.CTkFont(size=13)).pack(pady=(12, 8))

        # Preview da imagem
        self.label_img = ctk.CTkLabel(frame, text="Nenhuma imagem selecionada",
                                      width=200, height=200,
                                      fg_color=CINZA_CLARO, corner_radius=8)
        self.label_img.pack(pady=8)

        ctk.CTkButton(frame, text="📂 Carregar imagem",
                      command=self._carregar_imagem).pack(pady=4)

        # Resultado
        self.label_resultado = ctk.CTkLabel(frame, text="—",
                                            font=ctk.CTkFont(size=48, weight="bold"),
                                            text_color="steelblue")
        self.label_resultado.pack(pady=(16, 4))
        self.label_conf = ctk.CTkLabel(frame, text="",
                                       font=ctk.CTkFont(size=12), text_color="gray")
        self.label_conf.pack()

        ctk.CTkButton(frame, text="▶  Classificar",
                      command=self._classificar,
                      font=ctk.CTkFont(size=14, weight="bold"),
                      height=40).pack(pady=12)

        self._img_pil = None     # imagem carregada

    # ── Tab: Resultados ───────────────────────────────────────
    def _construir_tab_resultados(self):
        self.frame_graficos = self.tab_result
        ctk.CTkLabel(self.frame_graficos,
                     text="Treine e avalie a rede para ver os gráficos.",
                     text_color="gray").pack(pady=40)

    # ── Tab: Log ─────────────────────────────────────────────
    def _construir_tab_log(self):
        self.caixa_log = ctk.CTkTextbox(self.tab_log, font=ctk.CTkFont(family="Courier", size=12))
        self.caixa_log.pack(fill="both", expand=True, padx=8, pady=8)
        self._log("Sistema inicializado. Aguardando ações.\n")

    # ════════════════════════════════════════════════════════
    #  AÇÕES
    # ════════════════════════════════════════════════════════
    def _log(self, texto: str):
        self.caixa_log.configure(state="normal")
        self.caixa_log.insert("end", texto + "\n")
        self.caixa_log.see("end")
        self.caixa_log.configure(state="disabled")

    def _tentar_carregar_modelo(self):
        if os.path.exists(MODELO_PATH):
            try:
                self.rede = RedeAdaline.carregar(MODELO_PATH)
                self._log(f"Modelo carregado automaticamente: {MODELO_PATH}")
            except Exception as e:
                self._log(f"[AVISO] Não foi possível carregar modelo: {e}")

    def _gerar_imagens(self):
        self._log("Gerando dataset de imagens...")
        self.update()
        try:
            gerar_dataset(verbose=False)
            self._log("✅ Imagens geradas com sucesso em dados/treino e dados/teste")
        except Exception as e:
            self._log(f"❌ Erro ao gerar imagens: {e}")
            messagebox.showerror("Erro", str(e))

    def _treinar(self):
        """Executa o treinamento em thread separada para não travar a UI."""
        alpha    = float(self.var_alpha.get())
        res_str  = self.var_resolucao.get().split("×")[0]
        resolucao = int(res_str)
        epocas   = int(self.var_epocas.get())

        self._log(f"\nIniciando treinamento (α={alpha}, res={resolucao}px, épocas={epocas})...")

        def _worker():
            def _cb(letra, idx, total):
                self._log(f"  Neurônio '{letra}' treinado ({idx}/{total})")
                self.update_idletasks()

            try:
                self.rede = executar_treino(
                    taxa_aprendizado=alpha,
                    epocas=epocas,
                    resolucao=resolucao,
                    callback=_cb,
                )
                self._log("✅ Treinamento concluído!")
            except Exception as e:
                self._log(f"❌ Erro no treinamento: {e}")

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def _avaliar(self):
        if self.rede is None:
            messagebox.showwarning("Atenção", "Treine a rede primeiro!")
            return

        res_str   = self.var_resolucao.get().split("×")[0]
        resolucao = int(res_str)

        self._log("\nAvaliando no conjunto de teste...")
        try:
            resultado = calcular_acuracia(self.rede, resolucao=resolucao)
            self._log(f"  Acertos : {resultado['acertos']} / {resultado['total']}")
            self._log(f"  Acurácia: {resultado['global']:.2f}%")
            self._mostrar_graficos(resultado)
        except Exception as e:
            self._log(f"❌ Erro na avaliação: {e}")
            messagebox.showerror("Erro", str(e))

    def _mostrar_graficos(self, resultado: dict):
        """Renderiza os gráficos na aba Resultados."""
        for widget in self.frame_graficos.winfo_children():
            widget.destroy()

        ctk.CTkLabel(self.frame_graficos,
                     text=f"Acurácia global: {resultado['global']:.2f}%",
                     font=ctk.CTkFont(size=15, weight="bold")).pack(pady=(8, 4))

        # Gráfico convergência
        fig1 = figura_convergencia(self.rede)
        canvas1 = FigureCanvasTkAgg(fig1, master=self.frame_graficos)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill="x", padx=12, pady=4)

        # Gráfico acurácia por letra
        fig2 = figura_acuracia(resultado)
        canvas2 = FigureCanvasTkAgg(fig2, master=self.frame_graficos)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill="x", padx=12, pady=4)

        self.tabview.set("📊 Resultados")

    def _carregar_imagem(self):
        caminho = filedialog.askopenfilename(
            filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp")]
        )
        if not caminho:
            return
        self._img_pil = Image.open(caminho).convert("L")
        preview = ctk.CTkImage(light_image=self._img_pil, size=(180, 180))
        self.label_img.configure(image=preview, text="")
        self.label_img.image = preview
        self.label_resultado.configure(text="—")
        self.label_conf.configure(text="")

    def _classificar(self):
        if self.rede is None:
            messagebox.showwarning("Atenção", "Treine a rede primeiro!")
            return
        if self._img_pil is None:
            messagebox.showwarning("Atenção", "Carregue uma imagem primeiro!")
            return

        res_str   = self.var_resolucao.get().split("×")[0]
        resolucao = int(res_str)

        vetor  = pil_para_vetor(self._img_pil, resolucao=resolucao)
        letra  = self.rede.predizer(vetor)

        # Saídas lineares para mostrar "confiança"
        saidas = {l: self.rede.neuronios[l].saida_linear(vetor)
                  for l in self.rede.letras}
        top3   = sorted(saidas.items(), key=lambda kv: kv[1], reverse=True)[:3]
        conf   = " | ".join(f"{l}: {v:.2f}" for l, v in top3)

        self.label_resultado.configure(text=letra)
        self.label_conf.configure(text=f"Top-3 saídas lineares: {conf}")
        self._log(f"Classificação: {letra}  |  {conf}")


# ── Ponto de entrada ─────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()
