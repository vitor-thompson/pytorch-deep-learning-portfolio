# 🚀 Portfólio de Deep Learning com PyTorch

Bem-vindo ao meu repositório de estudos práticos em **Deep Learning com PyTorch**! Este espaço foi construído para consolidar e demonstrar meus conhecimentos na construção, treinamento, visualização e validação de Redes Neurais (NN) e Redes Neurais Convolucionais (CNN).

---

## 📂 Estrutura do Repositório

O projeto está dividido em duas trilhas principais na pasta **`notebooks/`**, acompanhadas de guias teóricos detalhados na pasta **`docs/`**:

### 🧠 Trilha 1: Fundamentos e Redes Neurais (NN)
| Parte | Tópico Abordado | Descrição Resumida |
| :--- | :--- | :--- |
| **01** | **Tensores no PyTorch** | Criação, manipulação e operações matemáticas fundamentais. |
| **02** | **Redes Neurais (NN)** | Construção de arquiteturas densas utilizando a API `torch.nn`. |
| **03** | **Treinamento de Redes** | Funções de perda (*Loss*), otimizadores e o algoritmo de *Backpropagation*. |
| **04** | **Fashion-MNIST** | Classificação de peças de vestuário com redes neurais. |
| **05** | **Inferência e Validação** | Avaliação de desempenho, cálculo de acurácia e prevenção de *overfitting*. |
| **06** | **Salvamento de Modelos** | Persistência de pesos e estados (`state_dict`) para reutilização. |
| **07** | **Carregamento de Imagens** | Uso de `ImageFolder` e pipelines de transformação (`transforms`). |
| **08** | **Transfer Learning** | Reaproveitamento de redes pré-treinadas para novos problemas. |

### 👁️ Trilha 2: Visão Computacional e Redes Convolucionais (CNN)
| Arquivo | Tópico Abordado | Descrição Resumida |
| :--- | :--- | :--- |
| **`mnist_mlp_exercise`** | **Baseline com MLP** | Classificação de dígitos manuscritos (MNIST) usando Perceptron Multicamadas como base de comparação. |
| **`custom_filters`** | **Filtros Customizados** | Aplicação manual de kernels (como filtros de Sobel) para detecção de bordas e padrões visuais. |
| **`conv_visualization`** | **Visualização de Convoluções** | Inspeção interna das camadas convolucionais e mapas de ativação para entender o que a rede enxerga. |

---

## 📖 Documentação Passo a Passo (`docs/`)

Para aprofundar o entendimento teórico por trás do código, acesse os guias explicativos na pasta **`docs/`**:
* [Guia 01: Entendendo Redes Neurais (NN) Passo a Passo](docs/01-redes-neurais-nn.md)
* [Guia 02: Como Funcionam as Redes Convolucionais (CNN)](docs/02-redes-convolucionais-cnn.md)

---
*Desenvolvido como portfólio técnico em Inteligência Artificial e Ciência de Dados.*
