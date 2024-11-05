# Análise de EEG usando Algoritmos Genéticos

## Sobre o Projeto
Este projeto implementa um sistema de análise de sinais de Eletroencefalograma (EEG) utilizando Algoritmos Genéticos para classificação de padrões neurológicos (simulações).

### Autores
- **Aluno:** João Renan Santanna Lopes
- **Professora:** Polyana Fonseca Nascimento
- **Instituição:** CESUPA (Centro Universitário do Estado do Pará)
- **Disciplina:** Algoritmos Genéticos

## Descrição
O sistema implementa uma interface gráfica interativa para análise e classificação de padrões de EEG relacionados a diferentes condições neurológicas:
- Normal
- Epilepsia
- Alzheimer
- Parkinson
- Depressão

### Funcionalidades Principais
1. **Geração de Sinais EEG**
   - Simulação de padrões específicos para cada condição
   - Visualização em tempo real do sinal
   - Análise espectral com diferentes bandas de frequência

2. **Algoritmo Genético**
   - População inicial aleatória
   - Seleção por torneio
   - Cruzamento uniforme
   - Mutação gaussiana
   - Visualização da evolução do fitness
   - Análise da distribuição populacional

3. **Interface Gráfica**
   - Controles interativos para parâmetros do AG
   - Múltiplas visualizações em tempo real
   - Exportação de resultados em Excel
   - Análise detalhada dos resultados

### Características Técnicas
- Extração de características do EEG:
  - Potência nas bandas Delta, Theta, Alpha e Beta
  - Características temporais
  - Medidas de complexidade
- Classificação genética com:
  - População configurável
  - Taxa de mutação ajustável
  - Número de gerações personalizável

## Requisitos
```python
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
scipy==1.11.3
scikit-learn==1.3.0
openpyxl==3.1.2
```

## Instalação
1. Clone o repositório
2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Uso
1. Execute o programa:
```bash
python main.py
```

2. Na interface:
   - Selecione a condição neurológica
   - Ajuste os parâmetros do AG
   - Gere um novo sinal
   - Inicie a evolução
   - Analise os resultados
   - Exporte os dados se necessário

## Estrutura do Projeto
- `AlgoritmoGeneticoEEG`: Implementação do AG
- `GeradorSinalEEG`: Simulação de sinais EEG
- `AnalisadorEEG`: Interface gráfica e visualizações

## Detalhes de Implementação
### Geração de Sinais
- Simulação de componentes de frequência específicos
- Adição de ruído controlado
- Padrões característicos para cada condição

### Algoritmo Genético
- Codificação: Vetor de pesos reais
- Fitness: Baseado no erro quadrático
- Seleção: Torneio (k=2)
- Cruzamento: Uniforme
- Mutação: Gaussiana

### Visualizações
1. **Sinal Temporal**
   - Visualização do EEG bruto
   - Marcações de eventos
   
2. **Evolução do AG**
   - Gráfico de fitness médio
   - Melhor fitness global
   
3. **Análise Populacional**
   - Distribuição dos genes
   - Boxplots por característica
   - Marcação do melhor indivíduo

## Contribuições e Melhorias Futuras
- Implementação de dados reais de EEG
- Adição de mais condições neurológicas
- Otimização dos parâmetros do AG
- Análise estatística mais detalhada
- Interface com banco de dados

## Licença
Este projeto é destinado apenas para fins educacionais e de pesquisa.

## Agradecimentos
Agradecimentos especiais à Professora Polyana Fonseca Nascimento pela orientação e ao CESUPA pelo suporte ao desenvolvimento deste projeto.

---
*Este projeto foi desenvolvido como parte da disciplina de Inteligência Artificial no CESUPA, 2024.*
