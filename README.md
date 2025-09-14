# Delivery Bot — README

Projeto do **DeliveryBot** referente à **AV1 da disciplina Inteligência Computacional – Universidade Cimatec**  
**Data de entrega:** 15/09/2025  

**Autores:**  
- Henrique Sá Barretto de Oliveira  
- Pedro Martins de Oliveira Menezes  

---

## 📦 Sobre o projeto

O **DeliveryBot** é um simulador de agente autônomo em uma planta industrial.  
Um robô percorre um **mapa em grade 30x30** com obstáculos, coletando pacotes e entregando-os em pontos de destino (metas) que surgem ao longo do tempo.  

O desempenho do agente é avaliado por um **sistema de pontuação** que recompensa entregas rápidas e pune passos desnecessários e atrasos.  

Este projeto permite:  
- Testar e comparar **estratégias de busca e navegação**.  
- Analisar cenários com obstáculos, restrições de tempo e múltiplos objetivos.  
- Aplicar conceitos de **logística interna** e **robótica móvel** em simulação.  

---

## ⚙️ 1) Instalar dependências

**Requisitos:**  
- Python 3.10+  
- Pygame  

Instalação:  

```bash
pip install pygame
```

---

## ▶️ 2) Executar o simulador

O jogo principal é rodado a partir de **main.py**, que já integra o **menu gráfico de configurações** (`menu.py`).

```bash
python main.py
```

No menu, escolha:  
- Agente (default, greedy, deadline, smart)  
- Seed (fixa aleatoriedade)  
- Sticky Target (habilita/desabilita)  
- Parâmetros extras (`max_carry`, `urgent_threshold`)  

---

## 💡 3) Dicas rápidas de uso

- `--agent` → escolha entre `default`, `greedy`, `deadline`, `smart`  
- `--seed` → fixa o mundo para repetir testes (ex.: `--seed 5`)  
- `--delay` → controla velocidade (ms entre passos)  
- `--sticky-target` → o robô não replaneja a cada passo (mantém alvo até chegar)  
- `--max-carry` → (apenas smart) quantos pacotes carregar antes de entregar  
- `--urgent-threshold` → (deadline/smart) define quando considerar meta “urgente”  

*(Esses parâmetros também podem ser configurados pelo menu gráfico)*  

---

## 🏆 4) Sistema de Pontuação

- **+50 pontos** por cada entrega concluída.  
- **–1 ponto** por cada passo dado.  
- **–1 ponto adicional por passo** para cada meta atrasada  
  (quando o tempo de vida/priority é ultrapassado).  

Essas regras já estão **100% implementadas no código**.  

---

## 🗺️ 5) Funcionamento do mapa e metas

- **7 pacotes** aparecem no mapa no início.  
- **6 metas** (pontos de entrega) surgem ao longo do tempo.  
- Cada meta possui uma **priority** (vida útil em passos).  
- Quando uma meta expira, começa a gerar penalidade.  
- O robô pode **carregar múltiplos pacotes** (limitado por `max_carry` no smart).  
- Obstáculos são gerados aleatoriamente (paredes horizontais/verticais + blocos sólidos).  
- Planejamento de caminho usa **A*** com heurística de **Manhattan**.  

---

## 🤖 6) Agentes implementados

### DefaultPlayer  
- Estratégia simples.  
- Pega pacote mais próximo → entrega na meta mais próxima.  
- Não considera prazos nem otimização.  

### GreedyBestFirst  
- Sempre busca o **alvo mais próximo** (pacote ou meta).  
- Vantagem: rápido, simples.  
- Desvantagem: ignora urgências → pode perder pontos.  

### DeadlineAwarePlayer  
- Considera prazos (`urgent_threshold`).  
- Priorização de metas com pouco tempo restante.  
- Evita penalidades pesadas de atrasos.  

### SmartBatchPlayer (original do grupo)  
- Estratégia mais avançada.  
- Combina capacidade (`max_carry`) + urgência (`urgent_threshold`).  
- Decide **quando coletar em lote** e **quando entregar já**.  
- Balanceia ganho líquido de pontos, penalidades e prazos.  

---

## 🖥️ 7) Exemplos de execução

Linha de comando:  

```bash
python main.py --agent default  --seed 2
python main.py --agent greedy   --seed 2
python main.py --agent deadline --seed 2 --urgent-threshold 10
python main.py --agent smart    --seed 2 --max-carry 2 --urgent-threshold 8 --delay 60
```

Loop automático de comparação (bash):  

```bash
for AG in default greedy deadline smart; do
  python main.py --agent $AG --seed 3 --delay 40
done
```

---

## 🔍 8) Scripts extras

Além do simulador principal (`main.py` + `menu.py`), o projeto inclui:  

### `batch_run.py`  
- Executa múltiplos testes automaticamente.  
- Varia agentes, seeds e parâmetros.  
- Salva resultados no CSV `resultados.csv`.  

### `analises_results.py`  
- Lê o CSV de resultados.  
- Faz análises comparativas (ex.: pontuação média, entregas concluídas, penalidades).  
- Auxilia na elaboração de gráficos e tabelas para o relatório/apresentação.  

---

## ⚙️ 9) Ajustes finos

- `--sticky-target` → fixa o alvo até alcançá-lo (pode evitar zigue-zague, mas prejudicar urgência).  
- `--max-carry` → aumenta/diminui a ganância do smart (quantos pacotes carregar antes de entregar).  
- `--urgent-threshold` → define urgência (quantos passos antes de expirar a meta).  

---

## 📊 10) Coleta e análise de resultados

1. Rodar o simulador várias vezes (diferentes agentes/seeds).  
2. O jogo salva o resultado final em `resultados.csv`.  
3. Usar `analises_results.py` para gerar estatísticas comparativas.  
4. Comparar desempenho dos agentes (eficiência, atrasos, pontuação final).  

---

## ✅ Observações finais

- O projeto respeita as regras originais do enunciado.  
- O **SmartBatchPlayer** foi uma criação do grupo (originalidade).  
- O jogo termina após **todas as 6 entregas concluídas**.  
- Se imagens (`images/cargo.png`, `images/operator.png`) não existirem, o jogo desenha blocos coloridos (fallback).  
- Todos os agentes usam **A\*** para planejar caminhos e evitar obstáculos.  
