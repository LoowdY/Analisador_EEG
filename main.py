import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import signal
import time
import pandas as pd
import random


class AlgoritmoGeneticoEEG:
    def __init__(self, num_caracteristicas=10, tamanho_pop=50, taxa_mutacao=0.1, taxa_cruzamento=0.8):
        self.num_caracteristicas = num_caracteristicas
        self.tamanho_pop = tamanho_pop
        self.taxa_mutacao = taxa_mutacao
        self.taxa_cruzamento = taxa_cruzamento
        self.populacao = []
        self.fitness_scores = []
        self.melhor_individuo = None
        self.melhor_fitness = float('-inf')
        self.historico_fitness = []
        self.historico_populacao = []
        self.geracao_atual = 0
        self.inicializar_populacao()

    def inicializar_populacao(self):
        self.populacao = [np.random.uniform(-1, 1, self.num_caracteristicas)
                          for _ in range(self.tamanho_pop)]

    def extrair_caracteristicas(self, sinal):
        f, pxx = signal.welch(sinal, fs=256)

        # Potência nas bandas de frequência
        delta = np.mean(pxx[(f >= 1) & (f <= 4)])
        theta = np.mean(pxx[(f >= 4) & (f <= 8)])
        alpha = np.mean(pxx[(f >= 8) & (f <= 13)])
        beta = np.mean(pxx[(f >= 13) & (f <= 30)])

        # Características temporais
        variancia = np.var(sinal)
        media = np.mean(np.abs(sinal))
        cruzamentos_zero = np.sum(np.diff(np.signbit(sinal)))
        pico_amplitude = np.max(np.abs(sinal))

        # Características de complexidade
        mobilidade = np.std(np.diff(sinal)) / np.std(sinal)
        complexidade = np.sqrt(
            np.var(np.diff(np.diff(sinal))) * np.var(sinal) /
            (np.var(np.diff(sinal)) ** 2)
        )

        return np.array([
            delta, theta, alpha, beta,
            variancia, media, cruzamentos_zero,
            pico_amplitude, mobilidade, complexidade
        ])

    def calcular_fitness(self, individuo, sinal, doenca_alvo):
        caracteristicas = self.extrair_caracteristicas(sinal)
        predicao = np.dot(caracteristicas, individuo)
        predicao = 1 / (1 + np.exp(-predicao))

        alvos = {
            'Normal': 0.1,
            'Epilepsia': 0.3,
            'Alzheimer': 0.5,
            'Parkinson': 0.7,
            'Depressão': 0.9
        }

        erro = (predicao - alvos[doenca_alvo]) ** 2
        return 1 / (1 + erro)

    def evoluir(self, sinal, doenca_alvo):
        self.fitness_scores = [self.calcular_fitness(ind, sinal, doenca_alvo)
                               for ind in self.populacao]

        melhor_idx = np.argmax(self.fitness_scores)
        if self.fitness_scores[melhor_idx] > self.melhor_fitness:
            self.melhor_fitness = self.fitness_scores[melhor_idx]
            self.melhor_individuo = self.populacao[melhor_idx].copy()

        self.historico_fitness.append(np.mean(self.fitness_scores))
        self.historico_populacao.append([ind.copy() for ind in self.populacao])

        # Seleção por torneio
        selecionados = []
        for _ in range(self.tamanho_pop):
            idx1, idx2 = random.sample(range(self.tamanho_pop), 2)
            if self.fitness_scores[idx1] > self.fitness_scores[idx2]:
                selecionados.append(self.populacao[idx1].copy())
            else:
                selecionados.append(self.populacao[idx2].copy())

        # Nova população
        nova_populacao = []
        while len(nova_populacao) < self.tamanho_pop:
            pai1, pai2 = random.sample(selecionados, 2)

            # Cruzamento
            if random.random() < self.taxa_cruzamento:
                ponto_corte = random.randint(1, self.num_caracteristicas - 1)
                filho = np.concatenate([pai1[:ponto_corte], pai2[ponto_corte:]])
            else:
                filho = pai1.copy()

            # Mutação
            if random.random() < self.taxa_mutacao:
                idx_mutacao = random.randint(0, self.num_caracteristicas - 1)
                filho[idx_mutacao] += random.gauss(0, 0.1)
                filho[idx_mutacao] = np.clip(filho[idx_mutacao], -1, 1)

            nova_populacao.append(filho)

        self.populacao = nova_populacao
        self.geracao_atual += 1

        return (np.mean(self.fitness_scores), np.max(self.fitness_scores),
                np.min(self.fitness_scores))


class CondicoesParada:
    def __init__(self):
        self.historico_fitness = []
        self.melhor_fitness_historico = []
        self.tolerancia = 1e-6
        self.geracoes_sem_melhoria = 0

    def convergencia_populacao(self, populacao, threshold=0.01):
        """Verifica se a população convergiu (indivíduos muito similares)"""
        fitness_std = np.std([np.std(ind) for ind in populacao])
        return fitness_std < threshold

    def estagnacao_melhor_fitness(self, fitness, num_geracoes=10):
        """Verifica se o melhor fitness não melhorou nas últimas N gerações"""
        self.melhor_fitness_historico.append(fitness)
        if len(self.melhor_fitness_historico) < num_geracoes:
            return False

        ultimas_n = self.melhor_fitness_historico[-num_geracoes:]
        melhoria = np.max(ultimas_n) - ultimas_n[0]
        return melhoria < self.tolerancia

    def estagnacao_media_fitness(self, fitness_medio, num_geracoes=10):
        """Verifica se a média do fitness não melhorou nas últimas N gerações"""
        self.historico_fitness.append(fitness_medio)
        if len(self.historico_fitness) < num_geracoes:
            return False

        ultimas_n = self.historico_fitness[-num_geracoes:]
        melhoria = np.max(ultimas_n) - ultimas_n[0]
        return melhoria < self.tolerancia

    def reset(self):
        """Reseta os históricos"""
        self.historico_fitness = []
        self.melhor_fitness_historico = []
        self.geracoes_sem_melhoria = 0

class GeradorSinalEEG:
    def __init__(self):
        self.fs = 256

    def gaussian_pulse(self, width, std):
        """Gera um pulso gaussiano manualmente"""
        x = np.linspace(-width/2, width/2, width)
        return np.exp(-(x**2) / (2*std**2))

    def gerar_sinal(self, doenca, duracao=1.0):
        t = np.linspace(0, duracao, int(self.fs * duracao))
        sinal = np.zeros_like(t)

        # Componentes base
        if doenca == 'Normal':
            # Alpha dominante posterior (8-13 Hz)
            sinal += 2.0 * np.sin(2 * np.pi * 10 * t)
            # Beta moderado (13-30 Hz)
            sinal += 0.5 * np.sin(2 * np.pi * 20 * t)

        elif doenca == 'Epilepsia':
            # Atividade de base
            sinal += 0.5 * np.sin(2 * np.pi * 3 * t)  # Delta aumentado
            # Adiciona espículas epileptiformes
            for _ in range(int(duracao * 3)):  # 3 espículas por segundo
                pos = np.random.randint(0, len(t))
                width = int(0.1 * self.fs)  # 100ms
                if pos + width < len(t):
                    # Usa nossa própria função gaussiana
                    spike = 3.0 * self.gaussian_pulse(width, width/8)
                    sinal[pos:pos + width] += spike

        elif doenca == 'Alzheimer':
            # Aumento de delta e theta
            sinal += 1.5 * np.sin(2 * np.pi * 2 * t)  # Delta
            sinal += 1.0 * np.sin(2 * np.pi * 6 * t)  # Theta
            # Redução de alpha
            sinal += 0.3 * np.sin(2 * np.pi * 10 * t)
            # Adiciona lentificação característica
            sinal *= (1 + 0.3 * np.sin(2 * np.pi * 0.5 * t))

        elif doenca == 'Parkinson':
            # Aumento de beta
            sinal += 2.0 * np.sin(2 * np.pi * 20 * t)
            # Tremor característico (4-6 Hz)
            sinal += 1.0 * np.sin(2 * np.pi * 5 * t)
            # Adiciona variabilidade do tremor
            tremor_freq = 5 + 0.5 * np.sin(2 * np.pi * 0.3 * t)
            sinal += 0.8 * np.sin(2 * np.pi * tremor_freq * t)

        elif doenca == 'Depressão':
            # Assimetria alpha frontal
            sinal += np.sin(2 * np.pi * 10 * t) * (1 + 0.3 * np.sin(2 * np.pi * 0.1 * t))
            # Redução geral de atividade
            sinal *= 0.7
            # Adiciona componente de baixa frequência
            sinal += 0.3 * np.sin(2 * np.pi * 0.2 * t)

        # Adiciona ruído fisiológico
        ruido = 0.1 * np.random.randn(len(t))
        # Modula o ruído
        ruido *= (1 + 0.2 * np.sin(2 * np.pi * 0.5 * t))
        sinal += ruido

        # Normaliza o sinal
        sinal = sinal / np.max(np.abs(sinal))

        return sinal, t


class AnalisadorEEG(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sistema de Análise de EEG com AG")
        self.geometry("1400x900")

        # Inicializa componentes
        self.gerador_sinal = GeradorSinalEEG()
        self.ag = AlgoritmoGeneticoEEG()
        self.sinal_atual = None
        self.tempo_atual = None
        self.evolucao_em_andamento = False

        # Cria interface
        self.criar_interface()

    def criar_interface(self):
        # Frame principal
        self.paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.paned.pack(expand=True, fill='both')

        # Frames laterais
        self.frame_controles = ttk.Frame(self.paned)
        self.frame_viz = ttk.Frame(self.paned)

        self.paned.add(self.frame_controles, weight=1)
        self.paned.add(self.frame_viz, weight=3)

        self.criar_controles()
        self.criar_visualizacoes()

    def criar_controles(self):
        # Parâmetros do AG
        frame_params = ttk.LabelFrame(self.frame_controles, text="Parâmetros do AG")
        frame_params.pack(fill='x', padx=5, pady=5)

        ttk.Label(frame_params, text="Tamanho da População:").pack(anchor='w')
        self.pop_size_var = tk.StringVar(value='50')
        ttk.Entry(frame_params, textvariable=self.pop_size_var).pack(fill='x', padx=5)

        ttk.Label(frame_params, text="Taxa de Mutação:").pack(anchor='w')
        self.mut_rate_var = tk.StringVar(value='0.1')
        ttk.Entry(frame_params, textvariable=self.mut_rate_var).pack(fill='x', padx=5)

        ttk.Label(frame_params, text="Número de Gerações:").pack(anchor='w')
        self.num_gen_var = tk.StringVar(value='50')
        ttk.Entry(frame_params, textvariable=self.num_gen_var).pack(fill='x', padx=5)

        # Seleção de doença
        frame_doenca = ttk.LabelFrame(self.frame_controles, text="Seleção de Condição")
        frame_doenca.pack(fill='x', padx=5, pady=5)

        self.doencas = ['Normal', 'Epilepsia', 'Alzheimer', 'Parkinson', 'Depressão']
        self.doenca_var = tk.StringVar(value='Normal')

        for doenca in self.doencas:
            ttk.Radiobutton(frame_doenca, text=doenca,
                            variable=self.doenca_var,
                            value=doenca).pack(anchor='w', padx=5)

        # Configuração temporal
        frame_tempo = ttk.LabelFrame(self.frame_controles, text="Configuração Temporal")
        frame_tempo.pack(fill='x', padx=5, pady=5)

        ttk.Label(frame_tempo, text="Duração (segundos):").pack(anchor='w')
        self.duracao_var = tk.StringVar(value='5')
        vcmd = (self.register(self.validar_duracao), '%P')
        self.duracao_entry = ttk.Entry(frame_tempo,
                                       textvariable=self.duracao_var,
                                       validate='key',
                                       validatecommand=vcmd)
        self.duracao_entry.pack(fill='x', padx=5)

        ttk.Label(frame_tempo, text="Janela de Visualização (segundos):").pack(anchor='w')
        self.janela_var = tk.StringVar(value='5')
        self.janela_entry = ttk.Entry(frame_tempo,
                                      textvariable=self.janela_var,
                                      validate='key',
                                      validatecommand=vcmd)
        self.janela_entry.pack(fill='x', padx=5)

        # Condições de Parada
        frame_parada = ttk.LabelFrame(self.frame_controles, text="Condições de Parada")
        frame_parada.pack(fill='x', padx=5, pady=5)

        # Checkbuttons para cada condição
        self.parada_geracoes = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame_parada, text="Número de Gerações",
                        variable=self.parada_geracoes).pack(anchor='w', padx=5)

        self.parada_convergencia = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame_parada, text="Convergência da População",
                        variable=self.parada_convergencia).pack(anchor='w', padx=5)

        self.parada_melhor = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame_parada, text="Estagnação do Melhor Fitness",
                        variable=self.parada_melhor).pack(anchor='w', padx=5)

        self.parada_media = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame_parada, text="Estagnação do Fitness Médio",
                        variable=self.parada_media).pack(anchor='w', padx=5)

        # Parâmetros das condições de parada
        ttk.Label(frame_parada, text="Gerações sem Melhoria:").pack(anchor='w')
        self.geracoes_estagnacao = tk.StringVar(value='10')
        ttk.Entry(frame_parada, textvariable=self.geracoes_estagnacao).pack(fill='x', padx=5)

        ttk.Label(frame_parada, text="Threshold Convergência:").pack(anchor='w')
        self.threshold_convergencia = tk.StringVar(value='0.01')
        ttk.Entry(frame_parada, textvariable=self.threshold_convergencia).pack(fill='x', padx=5)

        # Status
        frame_status = ttk.LabelFrame(self.frame_controles, text="Status")
        frame_status.pack(fill='x', padx=5, pady=5)

        self.label_geracao = ttk.Label(frame_status, text="Geração: 0")
        self.label_geracao.pack(anchor='w', padx=5)

        self.label_fitness = ttk.Label(frame_status, text="Melhor Fitness: 0.0")
        self.label_fitness.pack(anchor='w', padx=5)

        # Botões
        ttk.Button(self.frame_controles, text="Gerar Novo Sinal",
                   command=self.gerar_novo_sinal).pack(fill='x', padx=5, pady=5)

        ttk.Button(self.frame_controles, text="Iniciar Evolução",
                   command=self.iniciar_evolucao).pack(fill='x', padx=5, pady=5)

        self.btn_parar = ttk.Button(self.frame_controles, text="Parar Evolução",
                                    command=self.parar_evolucao, state='disabled')
        self.btn_parar.pack(fill='x', padx=5, pady=5)

        ttk.Button(self.frame_controles, text="Exportar Resultados",
                   command=self.exportar_resultados).pack(fill='x', padx=5, pady=5)

    def validar_duracao(self, valor):
        """Valida entrada de duração para aceitar apenas números positivos"""
        if valor == "":
            return True
        try:
            num = float(valor)
            return num > 0
        except ValueError:
            return False


    def criar_visualizacoes(self):
        """Cria todas as visualizações e gráficos"""
        # Notebook para abas
        self.notebook = ttk.Notebook(self.frame_viz)
        self.notebook.pack(expand=True, fill='both')

        # Aba do sinal temporal
        frame_temporal = ttk.Frame(self.notebook)
        self.notebook.add(frame_temporal, text="Sinal EEG")

        self.fig_temporal = Figure(figsize=(8, 3))
        self.ax_temporal = self.fig_temporal.add_subplot(111)
        self.canvas_temporal = FigureCanvasTkAgg(self.fig_temporal, frame_temporal)
        self.canvas_temporal.get_tk_widget().pack(fill='both', expand=True)

        # Aba do espectro
        frame_espectro = ttk.Frame(self.notebook)
        self.notebook.add(frame_espectro, text="Análise Espectral")

        self.fig_espectro = Figure(figsize=(8, 3))
        self.ax_espectro = self.fig_espectro.add_subplot(111)
        self.canvas_espectro = FigureCanvasTkAgg(self.fig_espectro, frame_espectro)
        self.canvas_espectro.get_tk_widget().pack(fill='both', expand=True)

        # Aba de características
        frame_carac = ttk.Frame(self.notebook)
        self.notebook.add(frame_carac, text="Características")

        self.fig_carac = Figure(figsize=(8, 3))
        self.ax_carac = self.fig_carac.add_subplot(111)
        self.canvas_carac = FigureCanvasTkAgg(self.fig_carac, frame_carac)
        self.canvas_carac.get_tk_widget().pack(fill='both', expand=True)

        # Aba de evolução do AG
        frame_evolucao = ttk.Frame(self.notebook)
        self.notebook.add(frame_evolucao, text="Evolução do AG")

        self.fig_evolucao = Figure(figsize=(8, 3))
        self.ax_evolucao = self.fig_evolucao.add_subplot(111)
        self.canvas_evolucao = FigureCanvasTkAgg(self.fig_evolucao, frame_evolucao)
        self.canvas_evolucao.get_tk_widget().pack(fill='both', expand=True)

        # Aba de população
        frame_pop = ttk.Frame(self.notebook)
        self.notebook.add(frame_pop, text="População")

        self.fig_pop = Figure(figsize=(8, 3))
        self.ax_pop = self.fig_pop.add_subplot(111)
        self.canvas_pop = FigureCanvasTkAgg(self.fig_pop, frame_pop)
        self.canvas_pop.get_tk_widget().pack(fill='both', expand=True)

        # Frame para estatísticas
        frame_stats = ttk.LabelFrame(self.frame_viz, text="Estatísticas")
        frame_stats.pack(fill='x', padx=5, pady=5)

        self.label_stats = ttk.Label(frame_stats, text="", justify='left')
        self.label_stats.pack(padx=5, pady=5)

        # Configuração inicial dos gráficos
        for ax in [self.ax_temporal, self.ax_espectro, self.ax_carac,
                   self.ax_evolucao, self.ax_pop]:
            ax.grid(True)

    def gerar_novo_sinal(self):
        try:
            duracao = float(self.duracao_var.get())
            self.sinal_atual, self.tempo_atual = self.gerador_sinal.gerar_sinal(
                self.doenca_var.get(), duracao=duracao)
            self.atualizar_visualizacoes()

            # Reinicializa o AG
            self.ag = AlgoritmoGeneticoEEG(
                tamanho_pop=int(self.pop_size_var.get()),
                taxa_mutacao=float(self.mut_rate_var.get())
            )

            self.label_geracao.config(text="Geração: 0")
            self.label_fitness.config(text="Melhor Fitness: 0.0")

        except ValueError as e:
            messagebox.showerror("Erro", "Valor de duração inválido!")

    def iniciar_evolucao(self):
        if self.sinal_atual is None:
            messagebox.showwarning("Aviso", "Gere um sinal primeiro!")
            return

        self.evolucao_em_andamento = True
        self.btn_parar.config(state='normal')

        try:
            num_geracoes = int(self.num_gen_var.get())
            doenca_alvo = self.doenca_var.get()
            geracoes_estagnacao = int(self.geracoes_estagnacao.get())
            threshold = float(self.threshold_convergencia.get())

            # Inicializa controlador de parada
            condicoes_parada = CondicoesParada()

            # Variáveis para tracking de resultados
            resultados = {
                'geracao_parada': 0,
                'motivo_parada': 'Número máximo de gerações atingido',
                'convergencia_atingida': False,
                'estagnacao_melhor': False,
                'estagnacao_media': False
            }

            for geracao in range(num_geracoes):
                if not self.evolucao_em_andamento:
                    resultados['motivo_parada'] = 'Parado pelo usuário'
                    break

                media_fitness, melhor_fitness, pior_fitness = self.ag.evoluir(
                    self.sinal_atual, doenca_alvo)

                # Verifica condições de parada
                parar = False

                if self.parada_convergencia.get():
                    if condicoes_parada.convergencia_populacao(self.ag.populacao, threshold):
                        parar = True
                        resultados['motivo_parada'] = "População convergiu"
                        resultados['convergencia_atingida'] = True

                if self.parada_melhor.get():
                    if condicoes_parada.estagnacao_melhor_fitness(melhor_fitness, geracoes_estagnacao):
                        parar = True
                        resultados['motivo_parada'] = "Melhor fitness estagnado"
                        resultados['estagnacao_melhor'] = True

                if self.parada_media.get():
                    if condicoes_parada.estagnacao_media_fitness(media_fitness, geracoes_estagnacao):
                        parar = True
                        resultados['motivo_parada'] = "Fitness médio estagnado"
                        resultados['estagnacao_media'] = True

                self.label_geracao.config(text=f"Geração: {geracao + 1}")
                self.label_fitness.config(text=f"Melhor Fitness: {melhor_fitness:.4f}")

                self.atualizar_grafico_evolucao()
                self.atualizar_grafico_populacao()

                self.update()
                time.sleep(0.1)

                if parar:
                    resultados['geracao_parada'] = geracao + 1
                    messagebox.showinfo("Evolução Finalizada",
                                        f"Algoritmo parou após {geracao + 1} gerações\n"
                                        f"Motivo: {resultados['motivo_parada']}")
                    break

                resultados['geracao_parada'] = geracao + 1

            self.btn_parar.config(state='disabled')
            self.evolucao_em_andamento = False

            if self.ag.melhor_individuo is not None:
                self.mostrar_resultados_finais(resultados)

        except Exception as e:
            messagebox.showerror("Erro", f"Erro durante a evolução: {str(e)}")
            self.evolucao_em_andamento = False
            self.btn_parar.config(state='disabled')

    def parar_evolucao(self):
            self.evolucao_em_andamento = False

    def atualizar_visualizacoes(self):
        """Atualiza todas as visualizações dos gráficos"""
        try:
            if self.sinal_atual is None:
                return

            # Atualiza gráfico temporal
            self.ax_temporal.clear()
            janela = float(self.janela_var.get())

            # Calcula índices para a janela de visualização
            amostras_por_janela = int(janela * self.gerador_sinal.fs)
            total_amostras = len(self.sinal_atual)

            if amostras_por_janela < total_amostras:
                self.ax_temporal.plot(self.tempo_atual[:amostras_por_janela],
                                      self.sinal_atual[:amostras_por_janela],
                                      'b-', linewidth=1)
            else:
                self.ax_temporal.plot(self.tempo_atual, self.sinal_atual,
                                      'b-', linewidth=1)

            self.ax_temporal.set_title(f"Sinal EEG - {self.doenca_var.get()}")
            self.ax_temporal.set_xlabel("Tempo (s)")
            self.ax_temporal.set_ylabel("Amplitude")
            self.ax_temporal.grid(True)

            # Atualiza análise espectral
            self.ax_espectro.clear()
            f, pxx = signal.welch(self.sinal_atual, fs=self.gerador_sinal.fs,
                                  nperseg=min(256, len(self.sinal_atual)))

            self.ax_espectro.semilogy(f, pxx, 'b-', linewidth=1)

            # Adiciona bandas de frequência
            for (fmin, fmax, cor, nome) in [
                (1, 4, 'red', 'Delta'),
                (4, 8, 'green', 'Theta'),
                (8, 13, 'blue', 'Alpha'),
                (13, 30, 'yellow', 'Beta')
            ]:
                mask = (f >= fmin) & (f <= fmax)
                self.ax_espectro.fill_between(f[mask], pxx[mask],
                                              alpha=0.3, color=cor, label=f'{nome} ({fmin}-{fmax} Hz)')

            self.ax_espectro.set_title("Análise Espectral")
            self.ax_espectro.set_xlabel("Frequência (Hz)")
            self.ax_espectro.set_ylabel("Densidade Espectral")
            self.ax_espectro.set_xlim(0, 40)
            self.ax_espectro.grid(True)
            self.ax_espectro.legend()

            # Atualiza características
            self.ax_carac.clear()

            # Extrai e normaliza características
            caracteristicas = [
                ('Delta', np.mean(pxx[(f >= 1) & (f <= 4)])),
                ('Theta', np.mean(pxx[(f >= 4) & (f <= 8)])),
                ('Alpha', np.mean(pxx[(f >= 8) & (f <= 13)])),
                ('Beta', np.mean(pxx[(f >= 13) & (f <= 30)])),
                ('Variância', np.var(self.sinal_atual)),
                ('Amplitude Pico', np.max(np.abs(self.sinal_atual))),
                ('Cruzamentos Zero', np.sum(np.diff(np.signbit(self.sinal_atual))) / len(self.sinal_atual)),
                ('Entropia', np.sum(np.abs(np.diff(self.sinal_atual)))),
                ('Mobilidade', np.std(np.diff(self.sinal_atual)) / np.std(self.sinal_atual)),
                ('Complexidade', np.log10(np.std(self.sinal_atual)))
            ]

            nomes = [c[0] for c in caracteristicas]
            valores = [c[1] for c in caracteristicas]

            # Normaliza valores para melhor visualização
            valores = np.array(valores)
            valores = (valores - np.min(valores)) / (np.max(valores) - np.min(valores))

            # Plota características
            barras = self.ax_carac.bar(range(len(caracteristicas)), valores)
            self.ax_carac.set_xticks(range(len(caracteristicas)))
            self.ax_carac.set_xticklabels(nomes, rotation=45, ha='right')

            # Adiciona valores sobre as barras
            for i, bar in enumerate(barras):
                height = bar.get_height()
                self.ax_carac.text(bar.get_x() + bar.get_width() / 2., height,
                                   f'{valores[i]:.2f}',
                                   ha='center', va='bottom', rotation=0)

            self.ax_carac.set_title("Características Normalizadas do Sinal")
            self.ax_carac.grid(True)

            # Atualiza estatísticas
            stats_text = (
                f"Estatísticas do Sinal:\n"
                f"Amplitude Máxima: {np.max(np.abs(self.sinal_atual)):.2f}\n"
                f"Média: {np.mean(self.sinal_atual):.2f}\n"
                f"Mediana: {np.median(self.sinal_atual):.2f}\n"
                f"Desvio Padrão: {np.std(self.sinal_atual):.2f}\n"
                f"RMS: {np.sqrt(np.mean(np.square(self.sinal_atual))):.2f}\n"
                f"Duração Total: {len(self.sinal_atual) / self.gerador_sinal.fs:.2f}s"
            )
            self.label_stats.config(text=stats_text)

            # Atualiza todos os canvas
            self.canvas_temporal.draw()
            self.canvas_espectro.draw()
            self.canvas_carac.draw()
            self.canvas_evolucao.draw()
            self.canvas_pop.draw()

        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao atualizar visualizações: {str(e)}")
            raise e

    def atualizar_grafico_evolucao(self):
            try:
                self.ax_evolucao.clear()

                if len(self.ag.historico_fitness) > 0:
                    geracoes = range(len(self.ag.historico_fitness))
                    self.ax_evolucao.plot(geracoes, self.ag.historico_fitness, 'b-', label='Média')

                    if self.ag.melhor_fitness != float('-inf'):
                        self.ax_evolucao.axhline(y=self.ag.melhor_fitness, color='r',
                                                 linestyle='--', label='Melhor Global')

                self.ax_evolucao.set_title("Evolução do Fitness")
                self.ax_evolucao.set_xlabel("Geração")
                self.ax_evolucao.set_ylabel("Fitness")
                self.ax_evolucao.legend()
                self.ax_evolucao.grid(True)
                self.canvas_evolucao.draw()

            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao atualizar gráfico de evolução: {str(e)}")

    def atualizar_grafico_populacao(self):
            try:
                self.ax_pop.clear()
                dados_pop = np.array(self.ag.populacao)

                if dados_pop.size > 0:
                    bp = self.ax_pop.boxplot(dados_pop.T)

                    if self.ag.melhor_individuo is not None:
                        x = range(1, len(self.ag.melhor_individuo) + 1)
                        self.ax_pop.plot(x, self.ag.melhor_individuo, 'r*',
                                         markersize=10, label='Melhor Indivíduo')

                self.ax_pop.set_title("Distribuição dos Genes na População")
                self.ax_pop.set_xlabel("Gene")
                self.ax_pop.set_ylabel("Valor")
                self.ax_pop.legend()
                self.ax_pop.grid(True)
                self.canvas_pop.draw()

            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao atualizar gráfico da população: {str(e)}")

    def exportar_resultados(self):
            if not hasattr(self.ag, 'historico_fitness') or len(self.ag.historico_fitness) == 0:
                messagebox.showwarning("Aviso", "Execute o AG primeiro!")
                return

            try:
                filename = filedialog.asksaveasfilename(
                    defaultextension=".xlsx",
                    filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
                )

                if filename:
                    # Cria DataFrames
                    evolucao_df = pd.DataFrame({
                        'Geração': range(len(self.ag.historico_fitness)),
                        'Fitness_Médio': self.ag.historico_fitness,
                        'Melhor_Fitness_Global': [self.ag.melhor_fitness] * len(self.ag.historico_fitness)
                    })

                    melhor_individuo_df = pd.DataFrame({
                        'Gene': [f'Gene_{i + 1}' for i in range(len(self.ag.melhor_individuo))],
                        'Valor': self.ag.melhor_individuo
                    })

                    params_df = pd.DataFrame({
                        'Parâmetro': ['Tamanho População', 'Taxa Mutação', 'Número Gerações', 'Doença'],
                        'Valor': [self.pop_size_var.get(), self.mut_rate_var.get(),
                                  self.num_gen_var.get(), self.doenca_var.get()]
                    })

                    # Salva em Excel
                    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                        evolucao_df.to_excel(writer, sheet_name='Evolução', index=False)
                        melhor_individuo_df.to_excel(writer, sheet_name='Melhor_Indivíduo', index=False)
                        params_df.to_excel(writer, sheet_name='Parâmetros', index=False)

                    messagebox.showinfo("Sucesso", "Resultados exportados com sucesso!")

            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao exportar resultados: {str(e)}")

    def mostrar_resultados_finais(self, resultados):
        try:
            janela = tk.Toplevel(self)
            janela.title("Resultados da Evolução")
            janela.geometry("400x600")

            frame = ttk.Frame(janela, padding="10")
            frame.pack(fill='both', expand=True)

            # Título
            ttk.Label(frame, text="Resultados da Classificação",
                      font=('Helvetica', 12, 'bold')).pack(pady=10)

            # Informações básicas
            ttk.Label(frame, text=f"Doença Simulada: {self.doenca_var.get()}",
                      font=('Helvetica', 10)).pack(pady=5)

            # Métricas da evolução
            ttk.Label(frame, text="\nMétricas de Evolução:",
                      font=('Helvetica', 10, 'bold')).pack(pady=5)

            ttk.Label(frame, text=f"Gerações Executadas: {resultados['geracao_parada']}").pack()
            ttk.Label(frame, text=f"Melhor Fitness: {self.ag.melhor_fitness:.4f}").pack()
            ttk.Label(frame, text=f"Fitness Médio Final: {self.ag.historico_fitness[-1]:.4f}").pack()

            # Condições de parada utilizadas
            ttk.Label(frame, text="\nCondições de Parada Utilizadas:",
                      font=('Helvetica', 10, 'bold')).pack(pady=5)

            if self.parada_geracoes.get():
                ttk.Label(frame, text=f"• Número máximo de gerações: {self.num_gen_var.get()}").pack()

            if self.parada_convergencia.get():
                status = "Atingida" if resultados['convergencia_atingida'] else "Não atingida"
                ttk.Label(frame,
                          text=f"• Convergência da população (threshold: {self.threshold_convergencia.get()}) - {status}").pack()

            if self.parada_melhor.get():
                status = "Atingida" if resultados['estagnacao_melhor'] else "Não atingida"
                ttk.Label(frame,
                          text=f"• Estagnação do melhor fitness ({self.geracoes_estagnacao.get()} gerações) - {status}").pack()

            if self.parada_media.get():
                status = "Atingida" if resultados['estagnacao_media'] else "Não atingida"
                ttk.Label(frame,
                          text=f"• Estagnação do fitness médio ({self.geracoes_estagnacao.get()} gerações) - {status}").pack()

            # Motivo da parada
            ttk.Label(frame, text="\nMotivo da Parada:",
                      font=('Helvetica', 10, 'bold')).pack(pady=5)
            ttk.Label(frame, text=resultados['motivo_parada']).pack()

            # Características do melhor indivíduo
            ttk.Label(frame, text="\nCaracterísticas do Melhor Indivíduo:",
                      font=('Helvetica', 10, 'bold')).pack(pady=5)

            tree = ttk.Treeview(frame, columns=('Gene', 'Valor'), show='headings', height=10)
            tree.heading('Gene', text='Gene')
            tree.heading('Valor', text='Valor')
            tree.pack(pady=10)

            for i, valor in enumerate(self.ag.melhor_individuo):
                tree.insert('', 'end', values=(f'Gene {i + 1}', f'{valor:.4f}'))

            # Botão para fechar
            ttk.Button(frame, text="Fechar", command=janela.destroy).pack(pady=10)

        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao mostrar resultados: {str(e)}")

if __name__ == "__main__":
    app = AnalisadorEEG()
    app.mainloop()
