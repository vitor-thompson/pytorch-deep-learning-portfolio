# 🎯 Guia Passo a Passo: Inferência, Validação e Overfitting (Parte 5)

Este documento documenta os conceitos avançados da quinta etapa do curso, abordando como avaliar o desempenho de modelos em dados desconhecidos por meio da **validação**, como identificar e mitigar o **overfitting** usando técnicas de **Dropout**, e como alternar corretamente entre os modos de treino e avaliação.

---

## 1. O Processo de Validação e Cálculo de Acurácia

Durante o treinamento, redes neurais podem se tornar excessivamente especializadas nos dados que já viram. Para medir a capacidade de generalização do modelo, utilizamos um **conjunto de validação/teste** separado.

* **Métrica de Acurácia:** Calculamos a porcentagem de previsões corretas comparando a classe mais provável gerada pelo modelo com os rótulos reais.
* **Método `.topk()`:** Para extrair a classe com maior probabilidade a partir das saídas da rede, utilizamos o método `.topk(1, dim=1)`, que retorna simultaneamente os maiores valores e os índices correspondentes.

### Exemplo do cálculo de acurácia por lote:
```python
# Converte log-probabilidades em probabilidades reais
ps = torch.exp(log_ps)

# Obtém a classe com maior probabilidade
top_p, top_class = ps.topk(1, dim=1)

# Compara as classes preditas com os rótulos reais ajustando os formatos
equals = top_class == labels.view(*top_class.shape)

# Calcula a média de acertos transformando o tensor booleano em float
accuracy = torch.mean(equals.type(torch.FloatTensor))
