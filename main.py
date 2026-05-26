# ============================================================
# main.py
# Ponto de entrada do projeto Adaline
# Execute: python main.py
# ============================================================

import os, sys

# Garante que a pasta raiz está no path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from interface import App

if __name__ == "__main__":
    app = App()
    app.mainloop()
