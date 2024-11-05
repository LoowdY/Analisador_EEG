# Análise de EEG com Algoritmos Genéticos

## Descrição
Este projeto implementa um sistema de análise de Eletroencefalograma (EEG) utilizando Algoritmos Genéticos para classificação de padrões neurológicos. O sistema permite a simulação, visualização e análise de sinais EEG para diferentes condições neurológicas (doenças).

### Desenvolvido por:
- **Aluno:** João Renan Santanna Lopes
- **Professora:** Polyana Fonseca Nascimento
- **Instituição:** CESUPA (Centro Universitário do Estado do Pará)
- **Disciplina:** Inteligência Artificial - Algoritmos Genéticos - Computação Evolutiva

## Funcionalidades

### 1. Geração de Sinais EEG
- Simulação de 5 padrões diferentes (doenças):
  - Normal
  - Epilepsia
  - Alzheimer
  - Parkinson
  - Depressão
- Duração configurável do sinal
- Janela de visualização ajustável

### 2. Análise Espectral
- Visualização das bandas de frequência:
  - Delta (1-4 Hz)
  - Theta (4-8 Hz)
  - Alpha (8-13 Hz)
  - Beta (13-30 Hz)
- Densidade espectral de potência
- Características espectrais normalizadas

### 3. Algoritmo Genético
- Parâmetros configuráveis:
  - Tamanho da população
  - Taxa de mutação
  - Número de gerações
- Visualização em tempo real da evolução
- Análise da população através de boxplots
- Tracking do melhor indivíduo

### 4. Análise Estatística
- Métricas do sinal:
  - Amplitude máxima
  - Média e mediana
  - Desvio padrão
  - RMS (Root Mean Square)
  - Duração total
- Características extraídas normalizadas
- Evolução do fitness ao longo das gerações

### 5. Exportação de Resultados
- Formato Excel (.xlsx)
- Múltiplas planilhas:
  - Evolução do AG
  - Melhor indivíduo
  - Parâmetros utilizados

## Requisitos
```python
numpy
pandas
matplotlib
scipy
tkinter
openpyxl
```

## Instalação
1. Clone o repositório
2. Instale as dependências:
```bash
pip install numpy pandas matplotlib scipy openpyxl
```

## Como Usar
1. Execute o programa:
```bash
python main.py
```

2. Na interface:
   - Configure os parâmetros do AG
   - Selecione a condição neurológica
   - Defina a duração do sinal
   - Ajuste a janela de visualização
   - Clique em "Gerar Novo Sinal"
   - Inicie a evolução do AG
   - Analise os resultados em tempo real
   - Exporte os resultados se necessário

## Estrutura do Projeto
- `AlgoritmoGeneticoEEG`: Implementação do AG
- `GeradorSinalEEG`: Simulação de sinais EEG
- `AnalisadorEEG`: Interface gráfica e visualizações

## Características Técnicas

### Geração de Sinais
- Frequência de amostragem: 256 Hz
- Componentes específicos para cada condição
- Ruído fisiológico modulado
- Normalização automática

### Algoritmo Genético
- Representação: Vetor de características
- Seleção: Torneio
- Cruzamento: Ponto único
- Mutação: Gaussiana
- Fitness baseado em erro quadrático

### Visualizações
1. **Sinal EEG**
   - Visualização temporal
   - Janela deslizante configurável

2. **Análise Espectral**
   - Densidade espectral de potência
   - Bandas de frequência destacadas

3. **Características**
   - 10 características normalizadas
   - Visualização em barras

4. **Evolução do AG**
   - Fitness médio por geração
   - Melhor fitness global

5. **População**
   - Distribuição dos genes
   - Melhor indivíduo destacado

## Contribuições
Este projeto faz parte de uma avaliação acadêmica e pode ser expandido com:
- Implementação de dados reais de EEG
- Adição de novos padrões neurológicos
- Melhorias na interface gráfica
- Otimização do algoritmo genético

## Licença
Este projeto é para fins educacionais e acadêmicos.

---
Desenvolvido como parte da disciplina de Inteligência Artificial no CESUPA, 2024.
