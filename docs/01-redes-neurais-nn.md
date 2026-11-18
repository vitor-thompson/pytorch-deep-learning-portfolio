# 🧠 Guia Passo a Passo: Fundamentos de Redes Neurais (NN) no PyTorch

Este documento detalha o raciocínio matemático e a implementação prática por trás das Redes Neurais Densas (também conhecidas como *Multi-Layer Perceptrons* - MLP) desenvolvidas nas primeiras etapas deste portfólio.

---

## 1. O Conceito de Tensor: A Bloco Construtor
No PyTorch, tudo gira em torno dos **Tensores** (`torch.Tensor`). Em termos simples:
* Um tensor de 0 dimensões é um número isolado (*Scalar*).
* Um tensor de 1 dimensão é uma lista de números (*Vector*).
* Um tensor de 2 dimensões é uma tabela (*Matrix*).
* Tensores de 3 dimensões ou mais representam lotes de imagens, dados em cubo, etc.

A grande vantagem dos tensores em relação às listas comuns do Python é que eles podem ser processados em paralelo utilizando a aceleração por hardware de placas gráficas (GPUs), o que é indispensável para o treinamento de redes neurais.

---

## 2. A Estrutura de uma Rede Neural Densa (`nn.Module`)

Para construir uma rede neural no PyTorch, criamos uma classe Python que herda de `torch.nn.Module`. Essa classe organiza duas partes fundamentais:
1. **O Construtor (`__init__`)**: Onde definimos as camadas (por exemplo, camadas lineares `nn.Linear`).
2. **O Método `forward`**: Onde definimos o caminho que os dados percorrem através da rede (o fluxo de cálculo).

### A Equação por Trás da Camada Linear
Cada neurônio em uma camada linear executa uma operação matemática fundamental:

$$y = W \cdot x + b$$

Onde:
* $x$ é a entrada (dados ou saída da camada anterior).
* $W$ são os **Pesos** (*Weights*), que determinam a importância de cada conexão.
* $b$ é o **Viés** (*Bias*), que ajusta o limiar de ativação do neurônio.

---

## 3. A Função de Ativação: Injetando Não-Linearidade
Se fizéssemos apenas multiplicações lineares sucessivas ($W \cdot x + b$), a rede neural inteira se comportaria como uma simples linha reta, independentemente de quantas camadas tivesse. 

Para permitir que a rede aprenda padrões complexos e curvas (como reconhecer formatos de dígitos ou roupas), aplicamos **funções de ativação não-lineares** logo após as camadas lineares. As mais comuns no curso foram:
* **ReLU (Rectified Linear Unit):** Zera todos os valores negativos ($f(x) = \max(0, x)$) e mantém os positivos inalterados. É a mais utilizada por sua eficiência computacional.
* **LogSoftmax / Softmax:** Usada tipicamente na camada final de problemas de classificação para transformar os outputs brutos (*logits*) em probabilidades que somam 1.

---

## 4. O Processo de Treinamento (O Coração do Aprendizado)

O treinamento de uma rede neural acontece em um ciclo iterativo composto por quatro passos principais:

1. **Forward Pass (Propagação para Frente):** Passamos os dados de entrada pela rede para obter uma predição.
2. **Cálculo da Perda (*Loss*):** Comparamos a predição da rede com o valor real esperado utilizando uma função de custo (como `nn.NLLLoss` ou `CrossEntropyLoss`). O resultado é um número que indica "o quão errado" o modelo está.
3. **Backward Pass (Retropropagação / Backpropagation):** Utilizando o sistema de diferenciação automática do PyTorch (`tensor.backward()`), calculamos o gradiente da perda em relação a cada peso da rede (a famosa derivada da física aplicada para achar o caminho de menor erro).
4. **Atualização dos Pesos (Otimizador):** O otimizador (como SGD ou Adam) atualiza os pesos na direção oposta ao gradiente, reduzindo o erro para a próxima rodada:
   
   $$W_{\text{novo}} = W_{\text{atual}} - (\text{taxa de aprendizado} \times \text{gradiente})$$

---

## 5. Resumo da Implementação Prática

No código do notebook correspondente (`Part 2 - Neural Networks in PyTorch`), a estrutura típica de um modelo se parece com isto:

```python
import torch
from torch import nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Definindo as camadas lineares
        self.hidden = nn.Linear(784, 256) # Entrada de 784 pixels (28x28) para 256 neurônios ocultos
        self.output = nn.Linear(256, 10)  # Saída para 10 classes possíveis (ex: dígitos de 0 a 9)
        
    def forward(self, x):
        # Achata a imagem se necessário e aplica a ativação ReLU na camada oculta
        x = F.relu(self.hidden(x))
        # Camada de saída com LogSoftmax para classificação
        x = F.log_softmax(self.output(x), dim=1)
        return x
