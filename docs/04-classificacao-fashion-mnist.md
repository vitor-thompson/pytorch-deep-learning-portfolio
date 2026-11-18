# 👕 Guia Passo a Passo: Classificação com Fashion-MNIST (Parte 4)

Este documento registra a aplicação prática dos conceitos de Deep Learning em um cenário mais desafiador do mundo real: o dataset **Fashion-MNIST**, composto por imagens de vestuário em tons de cinza ($28 \times 28$ pixels) divididas em 10 categorias distintas.

---

## 1. O Dataset Fashion-MNIST vs. MNIST Tradicional

Enquanto o dataset clássico de dígitos (MNIST) é relativamente simples e permite alcançar acurácias superiores a 97% com facilidade, o **Fashion-MNIST** apresenta maior complexidade visual (formas variadas de roupas, calçados e acessórios), servindo como um excelente parâmetro de desempenho para avaliar a robustez de redes neurais.

O carregamento e a separação em lotes (*batches*) utilizam o `torchvision`:

```python
from torchvision import datasets, transforms
import torch

# Normalização dos dados para o intervalo [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Baixando e carregando os conjuntos de treino e teste
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
