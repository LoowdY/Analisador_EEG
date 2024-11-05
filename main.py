import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import signal
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler
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


class GeradorSinalEEG:
    def __init__(self):
        self.fs = 256  # Frequência de amostragem

    def gerar_sinal(self, doenca, duracao=1.0):
        t = np.linspace(0, duracao, int(self.fs * duracao))
        sinal = np.zeros_like(t)

        if doenca == 'Normal':
            sinal += 2 * np.sin(2 * np.pi * 10 * t)  # Alpha dominante
            sinal += 0.5 * np.sin(2 * np.pi * 20 * t)  # Beta menor

        elif doenca == 'Epilepsia':
            sinal += np.sin(2 * np.pi * 25 * t)
            # Adiciona espículas
            for _ in range(3):
                pos = np.random.randint(0, len(t))
                sinal[pos:pos + 10] += 3 * np.sin(2 * np.pi * 50 * t[:10])

        elif doenca == 'Alzheimer':
            sinal += 0.5 * np.sin(2 * np.pi * 3 * t)  # Delta aumentado
            sinal += 0.3 * np.sin(2 * np.pi * 10 * t)  # Alpha reduzido

        elif doenca == 'Parkinson':
            sinal += 2 * np.sin(2 * np.pi * 20 * t)  # Beta aumentado
            sinal += 0.5 * np.sin(2 * np.pi * 10 * t)  # Alpha reduzido

        else:  # Depressão
            sinal += np.sin(2 * np.pi * 10 * t)  # Alpha assimétrico
            sinal += 0.3 * np.sin(2 * np.pi * 20 * t)  # Beta reduzido

        # Adiciona ruído
        sinal += 0.1 * np.random.randn(len(t))

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
        # Parâmetros
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

    def criar_visualizacoes(self):
        # Notebook para abas
        self.notebook = ttk.Notebook(self.frame_viz)
        self.notebook.pack(expand=True, fill='both')

        # Aba de sinal temporal
        frame_temporal = ttk.Frame(self.notebook)
        self.notebook.add(frame_temporal, text="Sinal EEG")

        self.fig_temporal = Figure(figsize=(8, 3))
        self.ax_temporal = self.fig_temporal.add_subplot(111)
        self.canvas_temporal = FigureCanvasTkAgg(self.fig_temporal, frame_temporal)
        self.canvas_temporal.get_tk_widget().pack(fill='both', expand=True)

        # Aba de evolução do AG
        frame_evolucao = ttk.Frame(self.notebook)
        self.notebook.add(frame_evolucao, text="Evolução do AG")

        self.fig_evolucao = Figure(figsize=(8, 6))
        self.ax_evolucao = self.fig_evolucao.add_subplot(111)
        self.canvas_evolucao = FigureCanvasTkAgg(self.fig_evolucao, frame_evolucao)
        self.canvas_evolucao.get_tk_widget().pack(fill='both', expand=True)

        # Aba de distribuição da população
        frame_pop = ttk.Frame(self.notebook)
        self.notebook.add(frame_pop, text="População")

        self.fig_pop = Figure(figsize=(8, 6))
        self.ax_pop = self.fig_pop.add_subplot(111)
        self.canvas_pop = FigureCanvasTkAgg(self.fig_pop, frame_pop)
        self.canvas_pop.get_tk_widget().pack(fill='both', expand=True)

    def gerar_novo_sinal(self):
        self.sinal_atual, self.tempo_atual = self.gerador_sinal.gerar_sinal(
            self.doenca_var.get())
        self.atualizar_visualizacoes()

        # Reinicializa o AG
        self.ag = AlgoritmoGeneticoEEG(
            tamanho_pop=int(self.pop_size_var.get()),
            taxa_mutacao=float(self.mut_rate_var.get())
        )

        self.label_geracao.config(text="Geração: 0")
        self.label_fitness.config(text="Melhor Fitness: 0.0")

    def iniciar_evolucao(self):
            if self.sinal_atual is None:
                messagebox.showwarning("Aviso", "Gere um sinal primeiro!")
                return

            self.evolucao_em_andamento = True
            self.btn_parar.config(state='normal')

            try:
                num_geracoes = int(self.num_gen_var.get())
                doenca_alvo = self.doenca_var.get()

                for geracao in range(num_geracoes):
                    if not self.evolucao_em_andamento:
                        break

                    media_fitness, melhor_fitness, pior_fitness = self.ag.evoluir(
                        self.sinal_atual, doenca_alvo)

                    self.label_geracao.config(text=f"Geração: {geracao + 1}")
                    self.label_fitness.config(text=f"Melhor Fitness: {melhor_fitness:.4f}")

                    self.atualizar_grafico_evolucao()
                    self.atualizar_grafico_populacao()

                    self.update()
                    time.sleep(0.1)

                self.btn_parar.config(state='disabled')
                self.evolucao_em_andamento = False

                if self.ag.melhor_individuo is not None:
                    self.mostrar_resultados_finais()

            except Exception as e:
                messagebox.showerror("Erro", f"Erro durante a evolução: {str(e)}")
                self.evolucao_em_andamento = False
                self.btn_parar.config(state='disabled')

    def parar_evolucao(self):
            self.evolucao_em_andamento = False

    def atualizar_visualizacoes(self):
            try:
                if self.sinal_atual is None:
                    return

                # Atualiza gráfico temporal
                self.ax_temporal.clear()
                self.ax_temporal.plot(self.tempo_atual, self.sinal_atual)
                self.ax_temporal.set_title("Sinal EEG")
                self.ax_temporal.set_xlabel("Tempo (s)")
                self.ax_temporal.set_ylabel("Amplitude")
                self.ax_temporal.grid(True)
                self.canvas_temporal.draw()

                # Limpa outros gráficos
                self.ax_evolucao.clear()
                self.ax_evolucao.set_title("Evolução do Fitness")
                self.ax_evolucao.set_xlabel("Geração")
                self.ax_evolucao.set_ylabel("Fitness")
                self.ax_evolucao.grid(True)
                self.canvas_evolucao.draw()

                self.ax_pop.clear()
                self.ax_pop.set_title("Distribuição da População")
                self.ax_pop.set_xlabel("Gene")
                self.ax_pop.set_ylabel("Valor")
                self.ax_pop.grid(True)
                self.canvas_pop.draw()

            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao atualizar visualizações: {str(e)}")

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

    def mostrar_resultados_finais(self):
            try:
                janela = tk.Toplevel(self)
                janela.title("Resultados da Evolução")
                janela.geometry("400x500")

                frame = ttk.Frame(janela, padding="10")
                frame.pack(fill='both', expand=True)

                ttk.Label(frame, text="Resultados da Classificação",
                          font=('Helvetica', 12, 'bold')).pack(pady=10)

                ttk.Label(frame, text=f"Doença Simulada: {self.doenca_var.get()}",
                          font=('Helvetica', 10)).pack(pady=5)

                ttk.Label(frame, text="\nMétricas de Evolução:",
                          font=('Helvetica', 10, 'bold')).pack(pady=5)

                ttk.Label(frame, text=f"Número de Gerações: {self.ag.geracao_atual}").pack()
                ttk.Label(frame, text=f"Melhor Fitness: {self.ag.melhor_fitness:.4f}").pack()
                ttk.Label(frame,
                          text=f"Fitness Médio Final: {self.ag.historico_fitness[-1]:.4f}").pack()

                ttk.Label(frame, text="\nCaracterísticas do Melhor Indivíduo:",
                          font=('Helvetica', 10, 'bold')).pack(pady=5)

                tree = ttk.Treeview(frame, columns=('Gene', 'Valor'), show='headings', height=10)
                tree.heading('Gene', text='Gene')
                tree.heading('Valor', text='Valor')
                tree.pack(pady=10)

                for i, valor in enumerate(self.ag.melhor_individuo):
                    tree.insert('', 'end', values=(f'Gene {i + 1}', f'{valor:.4f}'))

                ttk.Button(frame, text="Fechar", command=janela.destroy).pack(pady=10)

            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao mostrar resultados: {str(e)}")

if __name__ == "__main__":
    app = AnalisadorEEG()
    app.mainloop()
