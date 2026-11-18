# 🖼️ Guia Passo a Passo: Carregamento de Imagens e Data Augmentation (Parte 7)

Este documento detalha os conceitos da sétima etapa do curso, mostrando como lidar com imagens reais de maior resolução e múltiplos canais de cor (RGB) utilizando o dataset de cães e gatos (`Cat_Dog_data`), em vez de datasets simplificados como MNIST ou Fashion-MNIST.

---

## 1. Organizando e Carregando Imagens com `ImageFolder`

Quando trabalhamos com pastas de imagens no disco, a classe `datasets.ImageFolder` do `torchvision` facilita muito a leitura e a atribuição automática de rótulos. Ela espera que cada classe tenha sua própria subpasta:

```text
diretorio_raiz/
├── dog/
│   ├── imagem1.jpg
│   └── imagem2.jpg
└── cat/
    ├── imagem3.jpg
    └── imagem4.jpg
