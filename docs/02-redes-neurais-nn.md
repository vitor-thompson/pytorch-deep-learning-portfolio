# 🧬 Guia Passo a Passo: Redes Neurais com o Módulo `nn` do PyTorch (Parte 2)

Este documento documenta os conceitos da segunda etapa do curso, focando na transição de matrizes manuais para o uso do módulo avançado `torch.nn`, o processamento do dataset **MNIST** (dígitos manuscritos), a aplicação da função de ativação **Softmax** e a construção simplificada de arquiteturas via **`nn.Sequential`**.

---

## 1. Carregando e Transformando Dados com o MNIST

Para treinar redes capazes de reconhecer padrões complexos, utilizamos o conjunto de dados **MNIST**, composto por imagens em tons de cinza de $28 \times 28$ pixels representando números de 0 a 9.

Através do pacote `torchvision`, baixamos e preparamos os dados dividindo-os em lotes (*batches*):

```python
from torchvision import datasets, transforms
import torch

# Define transformações: converte para Tensor e normaliza os pixels para o intervalo [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Carregando os dados de treino
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
