# 📉 Guia Passo a Passo: Treinamento de Redes Neurais e Backpropagation (Parte 3)

Este documento documenta os conceitos essenciais da terceira etapa do curso, focando em como os modelos aprendem a partir dos dados: o cálculo de perdas (*Loss*), o mecanismo de diferenciação automática (*Autograd*), a retropropagação (*Backpropagation*) e a otimização dos pesos via gradiente descendente.

---

## 1. O Conceito de Função de Perda (*Loss*)

Uma rede neural recém-criada é "ingênua" e possui pesos aleatórios, gerando previsões incorretas. Para quantificar o erro do modelo, utilizamos uma **função de perda** (ou custo), que mede a distância entre a predição da rede e o rótulo verdadeiro.

* **Cross-Entropy Loss / NLLLoss:** Para problemas de classificação (como o MNIST), combinamos comumente a saída em log-probabilidades (`nn.LogSoftmax`) com a perda de verossimilhança negativa (`nn.NLLLoss`).
* Trabalhar com **log-probabilidades** evita problemas computacionais de precisão numérica associados a números muito próximos de 0 ou 1 gerados por probabilidades puras.

---

## 2. Diferenciação Automática com `autograd`

O PyTorch possui o módulo `autograd`, responsável por rastrear todas as operações realizadas em tensores que possuem `requires_grad=True`. 

* **Passagem para Frente (*Forward Pass*):** Os dados passam pelas camadas e calculam a perda.
* **Passagem para Trás (*Backward Pass*):** Ao executar `loss.backward()`, o PyTorch percorre o grafo de computação no sentido inverso aplicando a regra da cadeia (*chain rule* do cálculo) para calcular o gradiente da perda em relação a cada peso da rede.

> **Importante:** Os gradientes são acumulados a cada chamada de `loss.backward()`. Por isso, é obrigatório limpá-los no início de cada ciclo usando `optimizer.zero_grad()`.

---

## 3. O Otimizador (`optim`)

Com os gradientes calculados, precisamos de um algoritmo para atualizar os pesos na direção que reduz o erro (descida do gradiente). O pacote `torch.optim` fornece otimizadores eficientes como o SGD (*Stochastic Gradient Descent*):

```python
from torch import optim

# Definindo o otimizador com os parâmetros do modelo e a taxa de aprendizado (learning rate)
optimizer = optim.SGD(model.parameters(), lr=0.003)
