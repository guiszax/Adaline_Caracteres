# Reconhecimento de Caracteres com Adaline

Projeto da disciplina de Introdução à Inteligencia Artificial, ministrada pelo Prof. Fábio Oliveira

Instituto Federal de Brasília (IFB)

Alunos: Guilherme Souza e Pedro Neto

---

## Como rodar no VS Code

### 1. Pré-requisitos
- Possuir a ultimo versão instalada do Python
- VS Code com extensão **Python** (Microsoft)

### 2. Abrir o projeto
Abra a pasta `adaline_caracteres` no VS Code:
```
Arquivo → Abrir Pasta → selecione adaline_caracteres
```

### 3. Criar ambiente virtual
No terminal do VS Code (`Ctrl + '`):
```bash
python -m venv venv
```

Ativar:
- **Windows:** `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

### 4. Instalar dependências
```bash
pip install -r requirements.txt
```

### 5. Rodar o sistema
```bash
python main.py
```

---

## Fluxo de uso na interface

1. **Gerar imagens** → clique no botão "1️⃣ Gerar imagens"
2. **Treinar rede** → ajuste os parâmetros e clique "2️⃣ Treinar rede"
3. **Avaliar rede** → clique "3️⃣ Avaliar rede" para ver acurácia e gráficos
4. **Testar letra** → vá na aba "🔍 Testar letra", carregue uma imagem e clique "Classificar"

---

## Estrutura do projeto

```
adaline_caracteres/
├── dados/
│   ├── treino/        ← imagens geradas para treino
│   └── teste/         ← imagens geradas para teste
├── modelos/           ← modelo treinado (adaline.pkl)
├── resultados/        ← gráficos exportados
├── src/
│   ├── gerador.py        → gera imagens das letras
│   ├── processamento.py  → pré-processamento e vetorização
│   ├── adaline.py        → implementação da rede neural
│   ├── treino.py         → pipeline de treinamento
│   └── avaliacao.py      → métricas e gráficos
├── interface.py       ← interface gráfica (CustomTkinter)
├── main.py            ← ponto de entrada
├── requirements.txt
└── README.md
```

---

## Fontes utilizadas
- Arial (`arial.ttf`)
- Times New Roman (`times.ttf`)

---

## Parâmetros configuráveis na interface

| Parâmetro | Opções | Padrão |
|-----------|--------|--------|
| Taxa de aprendizado (α) | 0.1, 0.01, 0.001 | 0.01 |
| Resolução da imagem | 10×10, 20×20, 28×28 | 28×28 |
| Épocas máximas | (livre) | 200 |
