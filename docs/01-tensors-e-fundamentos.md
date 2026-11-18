# 📊 Guia Passo a Passo: Tensores e Fundamentos do PyTorch (Parte 1)

Este documento detalha os conceitos fundamentais do PyTorch abordados na primeira etapa do curso, focando na manipulação de tensores, na matemática por trás de um neurônio artificial e na interoperabilidade com o Numpy.

---

## 1. O que são Tensores?
No PyTorch, **tensores** (`torch.Tensor`) são a estrutura de dados central. Eles funcionam de maneira muito semelhante aos arrays do Numpy, com duas grandes vantagens:
* **Aceleração por Hardware:** Podem ser movidos facilmente para GPUs para processamento paralelo de alto desempenho.
* **Diferenciação Automática:** Possuem suporte nativo para calcular gradientes, essencial para o algoritmo de *Backpropagation*.

Visualmente, a dimensionalidade de um tensor funciona assim:
* **0 Dimensões:** Um número escalar isolado.
* **1 Dimensão:** Um vetor (ex: linha ou coluna de dados).
* **2 Dimensões:** Uma matriz (tabela de linhas e colunas).
* **3 Dimensões ou mais:** Dados complexos, como lotes de imagens coloridas (Canais x Altura x Largura).

---

## 2. Implementação: O Primeiro Neurônio e a Função de Ativação

O código inicial importa o PyTorch, define uma semente aleatória para garantir reprodutibilidade (`torch.manual_seed`) e cria dados simulados para entradas (*features*), pesos (*weights*) e viés (*bias*):

```python
import torch

# Função de ativação Sigmoid
def activation(x):
    """ Retorna a função sigmoide aplicada ao tensor x """
    return 1 / (1 + torch.exp(-x))

# Fixando a semente para resultados previsíveis
torch.manual_seed(7)

# Criando 5 variáveis de entrada aleatórias (1 linha, 5 colunas)
features = torch.randn((1, 5))

# Pesos com o mesmo formato das features
weights = torch.randn_like(features)

# Termo de viés (bias)
bias = torch.randn((1, 1))
