# 🔄 Guia Passo a Passo: Transfer Learning com Modelos Pré-Treinados (Parte 8)

Este documento detalha os conceitos da oitava etapa do curso, focando em **Transfer Learning** (transferência de aprendizado) para resolver problemas complexos de visão computacional — como classificar fotos de cães e gatos — aproveitando redes profundas já treinadas em milhões de imagens.

---

## 1. O que é Transfer Learning?

Treinar uma rede neural convolucional profunda do zero exige grandes volumes de dados e alto poder computacional. No **Transfer Learning**, pegamos um modelo já treinado em um dataset massivo (como o **ImageNet**, com mais de 1 milhão de imagens divididas em 1.000 categorias) e reutilizamos o conhecimento visual que ele já aprendeu (como detectar bordas, texturas e formas).

Essas camadas iniciais e intermediárias funcionam como excelentes **extratores de características** (*feature detectors*) para novas tarefas, mesmo que as imagens do novo problema não estejam no dataset original.

---

## 2. Pré-processamento e Normalização do ImageNet

Modelos pré-treinados do `torchvision.models` esperam entradas padronizadas:
* **Tamanho:** Imagens recortadas/redimensionadas para $224 \times 224$ pixels.
* **Normalização específica:** Cada canal de cor (R, G, B) deve ser normalizado com as mesmas médias e desvios padrão usados no treinamento original do ImageNet:
  * **Médias:** `[0.485, 0.456, 0.406]`
  * **Desvios padrão:** `[0.229, 0.224, 0.225]`

```python
from torchvision import datasets, transforms, models
import torch

data_dir = 'assets/Cat_Dog_data'

train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
