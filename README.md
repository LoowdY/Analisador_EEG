# Sistema de Análise de EEG com Algoritmos Genéticos
#### Disciplina: Inteligência Artificial
#### CESUPA - Centro Universitário do Estado do Pará
#### Professor: Polyana Fonseca Nascimento
#### Aluno: João Renan Santanna Lopes

## Sumário
1. [Introdução](#1-introdução)
2. [Fundamentação Teórica](#2-fundamentação-teórica)
3. [Metodologia](#3-metodologia)
4. [Implementação](#4-implementação)
5. [Resultados e Discussão](#5-resultados-e-discussão)
6. [Conclusão](#6-conclusão)
7. [Referências](#7-referências)

## 1. Introdução

### 1.1 Contextualização
O Eletroencefalograma (EEG) é uma ferramenta fundamental no diagnóstico e monitoramento de condições neurológicas. A análise automatizada desses sinais através de técnicas de Inteligência Artificial representa um avanço significativo na área médica, permitindo a identificação mais precisa de padrões associados a diferentes condições neurológicas.

### 1.2 Objetivos
- Desenvolver um sistema de análise de EEG utilizando Algoritmos Genéticos
- Implementar classificação automática de padrões neurológicos
- Simular e analisar sinais característicos de diferentes condições
- Avaliar a eficácia do AG na classificação dos padrões

## 2. Fundamentação Teórica

### 2.1 Eletroencefalograma (EEG)
O EEG registra a atividade elétrica cerebral através de eletrodos posicionados no couro cabeludo. Os principais ritmos cerebrais são:
- Delta (1-4 Hz): Sono profundo
- Theta (4-8 Hz): Sonolência
- Alpha (8-13 Hz): Relaxamento
- Beta (13-30 Hz): Estado de alerta

### 2.2 Algoritmos Genéticos
Inspirados na evolução biológica, os AGs utilizam:
- População de soluções candidatas
- Seleção natural
- Cruzamento e mutação
- Função de fitness para avaliação

### 2.3 Condições Neurológicas Analisadas
1. **Normal**
   - Ritmo alpha dominante
   - Atividade beta moderada

2. **Epilepsia**
   - Espículas epileptiformes
   - Atividade delta aumentada

3. **Alzheimer**
   - Lentificação do ritmo de base
   - Redução da atividade alpha

4. **Parkinson**
   - Aumento da atividade beta
   - Tremor característico 4-6 Hz

5. **Depressão**
   - Assimetria alpha frontal
   - Redução geral da atividade

## 3. Metodologia

### 3.1 Geração de Sinais
```python
def gerar_sinal(self, doenca, duracao=1.0):
    # Simulação dos padrões específicos
    # Componentes de frequência
    # Ruído fisiológico
```

### 3.2 Extração de Características
- Potência nas bandas de frequência
- Características temporais
- Medidas de complexidade

### 3.3 Algoritmo Genético
```python
class AlgoritmoGeneticoEEG:
    def __init__(self):
        # Inicialização
    
    def evoluir(self):
        # Seleção
        # Cruzamento
        # Mutação
```

### 3.4 Condições de Parada
1. Número máximo de gerações
2. Convergência da população
3. Estagnação do melhor fitness
4. Estagnação do fitness médio

## 4. Implementação

### 4.1 Componentes Principais
1. **Gerador de Sinais**
   - Simulação de padrões EEG
   - Características específicas por condição

2. **Algoritmo Genético**
   - Codificação: Vetor de características
   - Fitness: Erro quadrático
   - Operadores genéticos customizados

3. **Interface**
   - Visualização em tempo real
   - Controles interativos
   - Exportação de resultados

### 4.2 Parâmetros Configuráveis
- Tamanho da população
- Taxa de mutação
- Número de gerações
- Critérios de parada
- Duração do sinal

## 5. Resultados e Discussão

### 5.1 Desempenho do AG
- Taxa de convergência
- Tempo de processamento
- Precisão da classificação

### 5.2 Análise dos Sinais
- Características extraídas
- Padrões identificados
- Robustez do sistema

### 5.3 Limitações
- Simulação vs. dados reais
- Complexidade computacional
- Necessidade de ajuste de parâmetros

## 6. Conclusão

### 6.1 Contribuições
- Sistema integrado de análise
- Interface intuitiva
- Múltiplos critérios de parada
- Visualização em tempo real

### 6.2 Trabalhos Futuros
- Implementação com dados reais
- Otimização de parâmetros
- Análise estatística avançada
- Interface com banco de dados

## 7. Referências

1. Mitchell, M. (1998). An Introduction to Genetic Algorithms. MIT Press.
2. Niedermeyer, E., & da Silva, F. L. (2005). Electroencephalography: Basic Principles, Clinical Applications, and Related Fields. Lippincott Williams & Wilkins.
3. Holland, J. H. (1992). Adaptation in Natural and Artificial Systems. MIT Press.
4. Sanei, S., & Chambers, J. A. (2007). EEG Signal Processing. John Wiley & Sons.

---
**Observação**: Este projeto foi desenvolvido como parte da disciplina de Inteligência Artificial no CESUPA, 2024, sob orientação da Professora Polyana Fonseca Nascimento.
