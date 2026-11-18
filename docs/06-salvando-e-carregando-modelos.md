# 💾 Guia Passo a Passo: Salvando e Carregando Modelos no PyTorch (Parte 6)

Este documento documenta os conceitos da sexta etapa do curso, focando na persistência de modelos treinados. Em vez de retreinar uma rede neural toda vez que precisar usá-la, aprendemos a salvar os parâmetros e a arquitetura para reutilização em inferências futuras ou novos ciclos de treinamento.

---

## 1. O Dicionário de Estado (`state_dict`)

Os parâmetros de uma rede PyTorch (pesos e vieses de cada camada) são armazenados internamente em um dicionário chamado `state_dict`.

Podemos inspecionar as chaves do dicionário de estado de um modelo treinado da seguinte forma:

```python
print(model.state_dict().keys())
# Exemplo de saída: odict_keys(['hidden_layers.0.weight', 'hidden_layers.0.bias', ...])
