# 👁️ Guia Passo a Passo: Redes Neurais Convolucionais (CNN) e Visão Computacional

Este documento detalha o funcionamento das Redes Neurais Convolucionais (CNNs), explicando como computadores conseguem extrair características espaciais, detectar bordas e compreender imagens, indo além das redes densas tradicionais (MLP).

---

## 1. O Problema das Redes Densas (MLP) com Imagens

Nas redes neurais tradicionais (como as MLPs que você usou no MNIST), cada pixel da imagem é tratado como uma entrada isolada e totalmente conectada (`nn.Linear`). 
* **Perda espacial:** Quando achatamos uma imagem 2D em um vetor 1D, perdemos a relação de proximidade entre os pixels vizinhos.
* **Explosão de parâmetros:** Se pegarmos uma imagem colorida de tamanho moderado (ex: 224x224 pixels com 3 canais RGB), a primeira camada linear exigiria centenas de milhares de pesos apenas para o primeiro neurônio, tornando o modelo pesado e sujeito a *overfitting*.

---

## 2. A Camada Convolucional (`nn.Conv2d`)

As CNNs resolvem esse problema aplicando **filtros (kernels)** que deslizam por toda a imagem. Em vez de conectar todos os pixels de uma vez, a rede analisa pequenas regiões locais (janelas deslizantes).

### Como o Filtro Funciona (Convolução)
Um filtro é uma matriz pequena (ex: 3x3) de pesos que realiza uma operação de produto escalar elemento a elemento com a região da imagem sobre a qual está posicionado, gerando um **Mapa de Ativação** (*Feature Map*).

Dependendo dos valores dos pesos do filtro, a operação consegue destacar características específicas:
* **Filtros de Sobel (Detecção de Bordas):** Conforme explorado no notebook `custom_filters`, kernels específicos conseguem realçar variações bruscas de intensidade na imagem, identificando linhas verticais ou horizontais.
* **Filtros de Textura e Padrões:** Conforme a rede avança pelas camadas mais profundas (visualizado em `conv_visualization`), os filtros aprendem a reconhecer formas geométricas complexas, texturas e partes de objetos.

---

## 3. O Papel do Pooling (Redução Dimensional)

Logo após as camadas convolucionais, é comum aplicarmos camadas de agrupamento (*Pooling*, como `nn.MaxPool2d`). 
* O objetivo principal é **reduzir a dimensionalidade espacial** (altura e largura) dos mapas de características.
* O *Max Pooling*, por exemplo, pega uma janela (ex: 2x2) e mantém apenas o maior valor encontrado nela.
* Isso torna o modelo mais eficiente computacionalmente e garante **invariância à translação** (ou seja, se o objeto se mover um pouco na imagem, a rede ainda consegue reconhecê-lo).

---

## 4. Arquitetura Básica de uma CNN

Uma CNN típica é composta por duas grandes fases:

1. **Extrator de Características (Feature Extractor):** Sequência de camadas convolucionais (`Conv2d`) combinadas com funções de ativação (`ReLU`) e reduções (`MaxPool2d`). É onde a rede "aprende a enxergar".
2. **Classificador (Classifier):** Camadas densas tradicionais (`Linear`) no final da rede que recebem os mapas de características achatados e decidem a qual classe a imagem pertence (ex: gato, cachorro, dígito, peça de roupa).

---

## 5. Exemplo de Estrutura em Código PyTorch

Uma arquitetura simples de CNN inspirada nos seus estudos se estrutura assim:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Camada Convolucional 1: Entrada de 1 canal (ex: escala de cinza), 16 filtros de tamanho 3x3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        
        # Max Pooling para reduzir a imagem pela metade
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Camada Densa final para classificação (ex: 10 classes)
        # Se a imagem inicial for 28x28, após o pooling vira 14x14. Com 16 canais: 14 * 14 * 16 = 3136
        self.fc1 = nn.Linear(16 * 14 * 14, 10)

    def forward(self, x):
        # Passa pela convolução, aplica ReLU e depois o Pooling
        x = self.pool(F.relu(self.conv1(x)))
        
        # Achata o tensor para o formato de vetor aceito pela camada linear
        x = x.view(-1, 16 * 14 * 14)
        
        # Camada de saída
        x = self.fc1(x)
        return x
