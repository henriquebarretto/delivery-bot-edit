# Delivery Bot — README

Projeto do DeliveryBot referente a AV1 da disciplina Inteligência Computacional - Universidade Cimatec.
15/09/2025

Feito por:
Henrique Sá Barretto de Oliveira e 
Pedro Martins de Oliveira Menezes

O DeliveryBot é um simulador de agente autônomo ambientado em uma planta industrial. Nele, um robô deve percorrer um mapa em grade, coletando pacotes e entregando-os em pontos de destino que surgem ao longo do tempo. O desempenho do agente é medido por um sistema de pontuação que recompensa entregas rápidas e pune passos desnecessários e atrasos. O projeto permite testar estratégias de busca e navegação, além de comparar diferentes agentes autônomos em cenários com obstáculos, restrições de tempo e múltiplos objetivos, aproximando-se de desafios reais de logística interna e robótica móvel. Você pode modificar a simulação para torná-la mais realista ou complexa.

### 1) Instalar dependências

**Requisitos:**

- Python 3.10+

Instalar Pygame:

```bash
pip install pygame
```

### 2) Rodar o simulador

```bash
python main.py --agent smart --seed 5 --delay 60
```
** Mais em: Exemplos de execução

### 3) Dicas rápidas de uso

--agent pode ser default, greedy, deadline ou smart (recomendado).

--seed fixa o mundo para repetir o teste (ex.: --seed 5).

--delay controla a velocidade (ms) entre passos (ex.: --delay 60).

--sticky-target faz o robô não replanejar a cada passo (mantém o alvo até chegar).

### 4) Entender as regras (score)

- +50 por cada entrega concluída.

- –1 por cada passo (custo base).

- –1 extra por passo por cada entrega atrasada (quando a idade da meta ultrapassa sua prioridade/limite).

Essas regras estão implementadas no código: a cada passo há custo base e a função de penalidade conta metas atrasadas.

### 5) Como o mapa e metas funcionam

- Existem 7 pacotes gerados no mapa e 6 pontos de entrega no total (1 pacote a mais que entregas).

- As metas aparecem ao longo do tempo até totalizar 6 metas.

- O enunciado original sugere metas iniciais e spawns programados; o código gera metas ao longo do jogo para resultar em 6 entregas possíveis.

- Os tempos/intervalos são aleatórios por seed; veja Maze.__init__ para ajustar a forma/quantidade inicial de spawns se quiser reproduzir um comportamento diferente (ex.: 1 ou 2 metas no passo 0).

- Se uma meta é removida (entrega concluída), o ponto some do mapa.

- Você pode carregar mais de um pacote ao mesmo tempo (alguns agentes, como smart, aproveitam isso).

### 6) Escolher e comparar agentes

Rode múltiplas vezes (seeds diferentes / sempre a mesma) e faça a analise pelo arquivo csv de resultados gerado.
#### Tipos de agentes:

DefaultPlayer

- **Pega o pacote mais próximo** → entrega na meta mais próxima.
- Muito simples, não considera urgência nem capacidade.
- Vai bem em cenários “fáceis” (sem muitos prazos curtos).
- Sofre quando várias metas expiram rápido.

GreedyBestFirst (Ganancioso)

- **Sempre corre para o objeto/meta mais próximo.**
- Escolhe alvos pelo menor custo/distância (sem considerar prazos).
- Vantagem: rápido e simples, minimiza distância imediata.
- Desvantagem: pode ignorar urgências e ser penalizado quando metas expiram.
- Funciona bem em mapas densos de obstáculos, pois evita caminhos longos.

DeadlineAwarePlayer

- Prioriza metas com pouco tempo restante (usando o `urgent_threshold`).
- Vantagem: evita perder pontos por atrasos.
- Desvantagem: pode gastar muito tempo indo para metas distantes, deixando pacotes fáceis para trás.
- Bom quando você quer maximizar entregas dentro do prazo.


SmartBatchPlayer

- Tenta balancear capacidade (`max_carry`) e urgência(`urgent_threshold`).
- Coleta vários pacotes antes de sair entregando.
- Vantagem: mais eficiente em cenários com várias metas aparecendo em paralelo.
- Desvantagem: se o `max_carry` for muito alto ou o `threshold` mal calibrado, pode ficar "ganancioso" e perder prazos.
- Potencialmente o mais forte, mas precisa de tunagem de parâmetros.

##### Exemplos de execução:

    python main.py --agent default  --seed 2
    
    python main.py --agent greedy   --seed 2
    
    python main.py --agent deadline --seed 2 --urgent-threshold 10
    
    python main.py --agent smart    --seed 2 --max-carry 2 --urgent-threshold 8


### 7) Ajustes finos (Opcional)

--max-carry (apenas smart): quantos pacotes carregar antes de priorizar entregas.

--urgent-threshold (para deadline/smart): quantos passos faltando para considerar uma meta "urgente".

--sticky-target:

O sticky_target é um recurso que controla se o agente deve “grudar” no alvo escolhido até alcançá-lo ou se pode mudar de alvo a cada iteração quando aparece algo mais prioritário (ex: uma entrega prestes a expirar). Então, em quais agentes faz sentido?

- DefaultPlayer:

    Pode usar, mas o efeito é mínimo, ele já persegue sempre o objetivo mais próximo. Sticky só impede de mudar no meio do caminho.

- GreedyBestFirst:
    
    Funciona, mesmo raciocínio do default. Ele sempre pega o alvo mais próximo; sticky só “trava” a decisão até chegar.

- DeadlineAwarePlayer:
    
    Aqui fica mais interessante: sem sticky ele pode trocar de meta se uma entrega estiver prestes a expirar; com sticky ele vai até o alvo escolhido mesmo que outro mais urgente apareça → isso pode prejudicar, dependendo.

- SmartBatchPlayer:
    
    Mesmo caso do deadline. Como ele usa batching e thresholds, o sticky pode atrapalhar a lógica, já que a força dele está justamente em reavaliar constantemente.

### 8) Parâmetros de linha de comando (resumo)

    --agent : default / greedy / deadline / smart

    --seed : int (reprodutibilidade)

    --delay : int (ms entre passos)

    --sticky-target : flag (manter alvo até chegar)

    --max-carry : int (apenas smart)

    --urgent-threshold : int (apenas deadline/smart)