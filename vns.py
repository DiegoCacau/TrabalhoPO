import random
import numpy as np
import time

# Função para ler os dados do arquivo
def ler_dados_de_arquivo(arquivo):
    with open(arquivo, 'r') as f:
        linhas = f.readlines()

    dados = {}
    
    # Lendo o número de depósitos, clientes e veículos
    dados['num_depositos'] = int(linhas[0].strip())
    dados['num_clientes'] = int(linhas[1].strip())
    dados['num_veiculos'] = int(linhas[2].strip())
    
    # Lendo capacidades, custos fixos e variáveis dos veículos
    dados['capacidade_veiculos'] = list(map(int, linhas[3].strip().split()))
    dados['custo_fixo'] = list(map(int, linhas[4].strip().split()))
    dados['custo_variavel'] = list(map(int, linhas[5].strip().split()))
    
    # Lendo posições dos clientes e depósitos
    posicoes = [tuple(map(int, linha.strip().split())) for linha in linhas[6:]]
    dados['posicoes_clientes'] = posicoes[:dados['num_clientes']]
    dados['posicoes_depositos'] = posicoes[dados['num_clientes']:(dados['num_clientes'] + dados['num_depositos'])]

    return dados

# Lendo dados do arquivo de entrada
dados = ler_dados_de_arquivo('mdvrp_input.txt')

num_clientes = dados['num_clientes']
num_depositos = dados['num_depositos']
num_veiculos = dados['num_veiculos']
capacidade_veiculos = dados['capacidade_veiculos']
custo_fixo = dados['custo_fixo']
custo_variavel = dados['custo_variavel']
posicoes_clientes = {i: pos for i, pos in enumerate(dados['posicoes_clientes'])}
posicoes_depositos = {i: pos for i, pos in enumerate(dados['posicoes_depositos'])}

# Função para calcular a distância euclidiana entre dois pontos
def distancia_euclidiana(ponto1, ponto2):
    return int(((ponto1[0] - ponto2[0]) ** 2 + (ponto1[1] - ponto2[1]) ** 2) ** 0.5)

# Matriz de distâncias entre todos os nós (clientes e depósitos)
todos_nos = {**posicoes_clientes, **{num_clientes + k: v for k, v in posicoes_depositos.items()}}
distancias = np.zeros((num_clientes + num_depositos, num_clientes + num_depositos), dtype=int)

for i in range(num_clientes + num_depositos):
    for j in range(num_clientes + num_depositos):
        distancias[i, j] = distancia_euclidiana(todos_nos[i], todos_nos[j])

# Demandas por cliente no formato (cliente, demanda)
demandas = [(i, random.randint(1, 15)) for i in range(num_clientes)]

# Função para calcular o custo total
def calcular_custo_total(rotas, tipo_veiculos):
    """Calcula o custo total considerando todas as rotas."""
    custo_total = 0
    for deposito, rotas_deposito in enumerate(rotas):
        for rota in rotas_deposito:
            carga = sum(demanda for cliente, demanda in demandas if cliente in rota)
            veiculo = tipo_veiculos[deposito][tuple(rota)]
            custo_total += custo_fixo[veiculo] + custo_variavel[veiculo] * sum(
                distancias[rota[i]][rota[i + 1]] for i in range(len(rota) - 1)
            )
    return custo_total

def selecionar_veiculo(carga):
    """Seleciona o menor veículo que pode atender à carga, penalizando veículos maiores."""
    melhor_veiculo = None
    menor_custo = float('inf')

    for i, capacidade in enumerate(capacidade_veiculos):
        if capacidade >= carga:
            custo_veiculo = custo_fixo[i] + capacidade * 0.1  # Penalização adicional
            if custo_veiculo < menor_custo:
                menor_custo = custo_veiculo
                melhor_veiculo = i

    if melhor_veiculo is not None:
        return melhor_veiculo
    raise ValueError(f"Nenhum veículo pode atender à carga de {carga}.")

def construir_solucao_inicial():
    """Constrói uma solução inicial associando clientes aos depósitos com base nas demandas."""
    rotas = [[] for _ in range(num_depositos)]
    tipo_veiculos = [{} for _ in range(num_depositos)]

    for deposito in range(num_depositos):
        clientes_disponiveis = [cliente for cliente, _ in demandas]
        while clientes_disponiveis:
            rota_atual = []
            carga_atual = 0

            for cliente in clientes_disponiveis[:]:
                demanda_cliente = next(demanda for c, demanda in demandas if c == cliente)
                if carga_atual + demanda_cliente <= capacidade_veiculos[-1] and len(rota_atual) < 5:  # Limite de clientes
                    rota_atual.append(cliente)
                    carga_atual += demanda_cliente
                    clientes_disponiveis.remove(cliente)

            if rota_atual:
                veiculo = selecionar_veiculo(carga_atual)
                rotas[deposito].append(rota_atual)
                tipo_veiculos[deposito][tuple(rota_atual)] = veiculo

    return rotas, tipo_veiculos

def atualizar_tipo_veiculos(rotas):
    """Atualiza o dicionário de tipos de veículos para as rotas fornecidas."""
    tipo_veiculos = [{} for _ in range(len(rotas))]
    for deposito, rotas_deposito in enumerate(rotas):
        for rota in rotas_deposito:
            carga = sum(demanda for cliente, demanda in demandas if cliente in rota)
            veiculo = selecionar_veiculo(carga)
            tipo_veiculos[deposito][tuple(rota)] = veiculo
    return tipo_veiculos

def vns(rotas, tipo_veiculos, max_iter=100):
    """Implementa a meta-heurística Variable Neighborhood Search."""
    melhor_solucao = rotas
    melhor_tipo_veiculos = tipo_veiculos
    melhor_custo = calcular_custo_total(rotas, tipo_veiculos)

    for _ in range(max_iter):
        vizinhanca = gerar_vizinhos(melhor_solucao)
        if not vizinhanca:
            continue
        for vizinho in vizinhanca:
            tipo_atualizado = atualizar_tipo_veiculos(vizinho)
            custo = calcular_custo_total(vizinho, tipo_atualizado)
            if custo < melhor_custo:
                melhor_solucao, melhor_tipo_veiculos = vizinho, tipo_atualizado
                melhor_custo = custo
                break

    return melhor_solucao, melhor_tipo_veiculos, melhor_custo

def gerar_vizinhos(rotas):
    """Gera vizinhos alterando as rotas."""
    vizinhos = []
    for deposito, rotas_deposito in enumerate(rotas):
        for i in range(len(rotas_deposito)):
            if len(rotas_deposito[i]) > 1:
                rota_alterada = rotas_deposito[i][:]
                random.shuffle(rota_alterada)
                nova_rotas = rotas[:]
                nova_rotas[deposito] = rotas_deposito[:]
                nova_rotas[deposito][i] = rota_alterada
                vizinhos.append(nova_rotas)
    return vizinhos

def simulated_annealing(rotas, tipo_veiculos, max_iter=1000, temp_inicial=1000, alfa=0.99):
    """Implementa a meta-heurística Simulated Annealing."""
    melhor_solucao = rotas
    melhor_tipo_veiculos = tipo_veiculos
    melhor_custo = calcular_custo_total(rotas, tipo_veiculos)

    solucao_atual = rotas
    tipo_atual = tipo_veiculos
    custo_atual = melhor_custo

    temperatura = temp_inicial

    for _ in range(max_iter):
        vizinhanca = gerar_vizinhos(solucao_atual)
        if not vizinhanca:
            continue
        vizinho = random.choice(vizinhanca)
        tipo_atualizado = atualizar_tipo_veiculos(vizinho)
        custo_vizinho = calcular_custo_total(vizinho, tipo_atualizado)

        delta = custo_vizinho - custo_atual
        if delta < 0 or random.random() < np.exp(-delta / temperatura):
            solucao_atual, tipo_atual, custo_atual = vizinho, tipo_atualizado, custo_vizinho

            if custo_vizinho < melhor_custo:
                melhor_solucao, melhor_tipo_veiculos, melhor_custo = vizinho, tipo_atualizado, custo_vizinho

        temperatura *= alfa

    return melhor_solucao, melhor_tipo_veiculos, melhor_custo

# Execução do algoritmo VNS
rotas_iniciais, tipos_iniciais = construir_solucao_inicial()

inicio_vns = time.time()
melhor_solucao_vns, melhor_tipo_veiculos_vns, melhor_custo_vns = vns(rotas_iniciais, tipos_iniciais)
tempo_vns = time.time() - inicio_vns

print("Dados de entrada:")
print("Número de clientes:", num_clientes)
print("Número de depósitos:", num_depositos)
print("Número de veículos:", num_veiculos)
print("Capacidades dos veículos:", capacidade_veiculos)
print("Custo fixo dos veículos:", custo_fixo)
print("Custo variável dos veículos:", custo_variavel)
print("Demandas por cliente:")
print(demandas)
print("Distâncias entre os nós:")
print(distancias)

print("\nResultados VNS:")
print("Melhor custo encontrado (VNS):", melhor_custo_vns)
print("Tempo de execução (VNS): {:.4f} segundos".format(tempo_vns))
for deposito, rotas in enumerate(melhor_solucao_vns):
    for rota in rotas:
        veiculo = melhor_tipo_veiculos_vns[deposito][tuple(rota)]
        print(f"Depósito {deposito}, Rota {rota}, Veículo utilizado: {veiculo}")

# Execução do algoritmo Simulated Annealing
inicio_sa = time.time()
melhor_solucao_sa, melhor_tipo_veiculos_sa, melhor_custo_sa = simulated_annealing(rotas_iniciais, tipos_iniciais)
tempo_sa = time.time() - inicio_sa + tempo_vns

print("\nResultados Simulated Annealing:")
print("Melhor custo encontrado (SA):", melhor_custo_sa)
print("Tempo de execução (SA): {:.4f} segundos".format(tempo_sa))
for deposito, rotas in enumerate(melhor_solucao_sa):
    for rota in rotas:
        veiculo = melhor_tipo_veiculos_sa[deposito][tuple(rota)]
        print(f"Depósito {deposito}, Rota {rota}, Veículo utilizado: {veiculo}")
