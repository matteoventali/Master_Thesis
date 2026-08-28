# Lunar Lander with LTLf Reward Shaping

Repository di lavoro per esperimenti di reinforcement learning su LunarLander
con reward shaping guidato da automi.

## Framework attivi

La repository mantiene due soli framework:

| Directory | Uso |
| --- | --- |
| `multilevel_framework` | Framework multi-livello con un singolo learner |
| `manual_experiment` | Framework manuale per task ciclici |

Ogni framework contiene i sorgenti in `src/`, i launcher Docker nella propria
root e scrive i nuovi artefatti in `results/<experiment-name>/`.

## Avvio

Dalla root della repository:

```bash
./multilevel_framework/run_experiment.sh \
  --experiment-name multilevel-base \
  --episodes 1000 --num-seeds 5 --seed 42

./manual_experiment/run_experiment.sh \
  --experiment-name manual-cycle \
  --episodes 1000 --num-seeds 5 --seed 42
```

Nel `multilevel_framework`, il target con aggiornamento stocastico di Bellman
e alpha è opzionale e disabilitato di default. Si abilita aggiungendo:

```bash
--stochastic-bellman-update --bellman-alpha 0.1
```

I launcher costruiscono l'immagine Docker del framework, verificano la
disponibilita di CUDA e montano la directory del framework in `/workspace`.
Gli script `run_evaluation.sh` usano lo stesso ambiente per valutare policy
gia addestrate.

## Configurazioni e notebook

Le configurazioni riutilizzabili sono raccolte in `templates/`:

- `templates/abstractions/` per le gerarchie multi-livello;
- `templates/trajectories/` per i task sequenziali;
- `templates/cyclic/` per i task ciclici.

I task definiscono ogni proposizione come una regione circolare continua nelle
coordinate normalizzate di LunarLander (`x` in `[-1, 1]`, `y` in `[0, 1.5]`):

```json
"regions": {
  "goal": {"center": [0.42, 1.05], "radius": 0.12}
}
```

`predicates` permette di definire tre tipi di proprietà spaziali atomiche nelle
coordinate dell'environment: `circle`, `box` e `half_plane`. Le regioni
circolari già presenti in `regions` restano compatibili e sono adatte a waypoint
e goal.

```json
{
  "formula": "F(wp1 & X(F(g1))) & G(wp1 -> X(G(!above_limit)))",
  "regions": {
    "wp1": {"center": [-0.4, 0.8], "radius": 0.1},
    "g1": {"center": [0.6, 1.0], "radius": 0.1}
  },
  "predicates": {
    "above_limit": {
      "type": "half_plane",
      "axis": "y",
      "operator": ">",
      "threshold": 0.9
    }
  }
}
```

Un cerchio e un rettangolo possono essere definiti in questo modo:

```json
"near_point": {
  "type": "circle",
  "center": [0.2, 0.7],
  "radius": 0.1
},
"corridor": {
  "type": "box",
  "x_min": -0.5,
  "x_max": 0.5,
  "y_min": 0.2,
  "y_max": 1.0
}
```

Per `half_plane`, `axis` può essere `x` oppure `y` e `operator` può essere `<`,
`<=`, `>` oppure `>=`; ogni predicato richiede una `threshold` numerica. Per
esempio:

```json
"below_limit": {
  "type": "half_plane",
  "axis": "y",
  "operator": "<",
  "threshold": 0.25
}
```

La dipendenza temporale rimane nella formula LTLf: `above_limit` significa
solamente che la posizione corrente ha `y > 0.9`, mentre
`G(wp1 -> X(G(!above_limit)))` attiva il vincolo dopo `wp1`. Sullo stato
continuo ogni predicato viene valutato sulla posizione esatta. Sul livello
astratto una cella soddisfa un predicato se interseca il cerchio, il rettangolo
o il semipiano corrispondente.

Training e valutazione verificano l'appartenenza sullo stato continuo. Per il
planning astratto, ogni regione etichetta tutte le celle che interseca; ogni
livello della gerarchia viene rasterizzato direttamente dalla regione continua.

Nel `multilevel_framework`, tutti i livelli non-top usano automaticamente il
dual Q-learning. Sul livello top si può scegliere tra `"value_iteration"` e
Q-learning classico (`"learning"`)
senza shaping. Gli iperparametri si configurano nel file di astrazione:

```json
{
  "name": "level1",
  "grid_w": 12,
  "grid_h": 12,
  "algorithm": "learning",
  "value_function_method": "policy_evaluation",
  "learning": {
    "episodes": 10000,
    "max_steps": 100,
    "alpha": 0.1,
    "epsilon_start": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.999,
    "gamma_shaping": 0.99,
    "seed": 0,
    "log_interval": 1000,
    "eval_interval": 10000,
    "eval_episodes": 500,
    "eval_seed": 100000
  }
}
```

I livelli sono elencati dal più fine al più grossolano e risolti in ordine
inverso. Tutti i livelli sottostanti al top usano una Q-table biased per
l'esplorazione con PBRS e una Q-table unbiased per il valore trasferito al
livello inferiore. Il top usa `"value_iteration"` per default; impostando `"algorithm":
"learning"` usa invece una sola Q-table con la reward originale. Questo vale
anche per una gerarchia con un solo livello. Se la sezione `learning` manca
vengono usati i valori predefiniti mostrati sopra.
Per ogni livello appreso, `value_function_method` seleziona come ricavare la
V-function unbiased dalla Q-table: `"max"` usa
`V(s) = max_a Q_unbiased(s,a)`, mentre `"policy_evaluation"` valuta a orizzonte
infinito scontato la policy greedy deterministica indotta da `Q_unbiased`. Il
default, mantenuto per compatibilita, e `"max"`.

Un livello gia appreso puo essere riutilizzato indicando il suo archivio come
checkpoint:

```json
{
  "name": "level1",
  "grid_w": 12,
  "grid_h": 12,
  "algorithm": "learning",
  "checkpoint": "../previous_experiment/results/abstract_value_functions/level1/value_function.npz"
}
```

Il percorso relativo viene risolto rispetto ad `abstraction.json`. Quando il
checkpoint e presente, Q-learning e costruzione della V vengono saltati e il
livello carica direttamente `q_function_unbiased` e
`v_function_unbiased`. Non viene verificato l'allineamento semantico del
checkpoint con regioni, reward o task correnti.
Negli stati DFA accettanti l'unica azione disponibile è `done`: assegna il
`goal_reward` e termina senza bootstrap. Il valore del goal viene quindi
appreso o calcolato dalla VI, senza inizializzare manualmente le Q-table.
Gli stati DFA dai quali non e piu raggiungibile alcuna accettazione sono
classificati automaticamente come fallimenti terminali. Ricevono task reward
zero e terminano senza bootstrap; l'eventuale shaping
`gamma_shaping * Phi(next) - Phi(state)` viene comunque applicato normalmente.

Per ogni livello appreso vengono salvati `reward_epsilon.png` e i relativi dati
in `reward_epsilon_data.npz`, sotto
`results/<esperimento>/img/abstract_learning/levelN/`. Il grafico mostra gli
andamenti delle reward biased e unbiased insieme a epsilon.
Per ogni livello, indipendentemente dall'algoritmo VI o learning, la V-function
numerica viene inoltre salvata in
`results/<esperimento>/results/abstract_value_functions/levelN/value_function.npz`.
Le chiavi `v_function_unbiased`, `unbiased_values` e, per i livelli dual,
`biased_values` usano l'ordinamento `[q_index, y, x]`; `values` rimane un alias
dell'unbiased per compatibilita. Per i livelli appresi viene salvata anche
`q_function_unbiased`, ordinata come `[q_index, y, x, action]`, rendendo lo
stesso file utilizzabile come checkpoint. `dfa_states` associa ogni `q_index` all'identificatore reale
dello stato DFA; il file include anche dimensioni, gamma, goal reward,
algoritmo risolutivo e il flag `has_biased_values`.
Questo salvataggio viene prodotto anche con `--no-heatmaps`.
Le V-function salvate possono essere confrontate con lo script
`multilevel_framework/compare_value_functions.py`. Avviandolo senza argomenti
si aprono due finestre: la prima seleziona il `value_function.npz` ottenuto con
VI, usato come riferimento, e la seconda seleziona quello ottenuto dal
Q-learning. Lo script confronta sia la V unbiased sia, quando presente, la V
biased, e salva metriche CSV/JSON, mappe dei valori e degli errori e grafici del
segnale di shaping:

```bash
~/.env/bin/python multilevel_framework/compare_value_functions.py
```

La stessa analisi puo essere eseguita senza finestre grafiche specificando i
file direttamente:

```bash
~/.env/bin/python multilevel_framework/compare_value_functions.py --reference /path/to/vi/value_function.npz --candidate /path/to/learning/value_function.npz
```

`trajectory.json` viene sempre richiesto con una terza finestra, in modo che la
formula e le regioni usate nel confronto siano scelte esplicitamente. In uso
non interattivo puo essere fornito tramite `--trajectory`. Gli output vengono creati nella directory
`value_function_comparison` dell'esperimento candidato, oppure nel percorso
indicato con `--output-dir`.
Ogni Q-learning astratto scrive inoltre un log in
`results/<esperimento>/logs/abstract_learning/levelN.log`; `log_interval`
controlla ogni quanti episodi vengono riportati reward medie recenti, success
rate sui restart non-goal, epsilon, aggiornamenti e tempo trascorso.
I restart che partono già in uno stato DFA accettante continuano ad addestrare
l'azione `done`, ma sono esclusi da success rate e curve reward per evitare il
plateau artificiale dovuto al reward immediato.
Ogni `eval_interval` episodi vengono eseguite due valutazioni greedy sugli
stessi start determinati da `eval_seed`. La prima evaluation parte da
posizioni casuali e stati DFA casuali non accettanti dai quali l'accettazione e
ancora raggiungibile, riporta i risultati anche per stato DFA iniziale e misura
la capacita di recupero durante le diverse fasi del task. La seconda evaluation
campiona le posizioni ma parte sempre dallo stato iniziale del DFA e
misura il completamento dell'intera formula. Nei livelli dual il log identifica
esplicitamente la shaping-guided biased Q e la original-reward unbiased Q; le serie aggregate sono
salvate anche in `reward_epsilon_data.npz`. Errore TD e numero di coppie
positive permettono di monitorare la propagazione della Q unbiased anche prima
che la policy completi il task.
Nei livelli non-top `gamma_shaping` controlla lo shaping ricevuto dal livello
superiore secondo `gamma_shaping_i * Phi_(i+1)(next) - Phi_(i+1)(state)`. Se
omesso viene usato il `gamma` dell'MDP. Sul livello top non è ammesso, perché
non esiste un potenziale superiore.
La generazione delle heatmap dei potenziali può essere disabilitata aggiungendo
`--no-heatmaps` al comando di training o post-processing; i grafici
reward–epsilon e i log astratti vengono comunque prodotti. Le heatmap non
mostrano annotazioni nelle celle per default. Con `--heatmap-annotation`, le
celle compatibili mostrano il valore numerico; quelle il cui ingresso cambia
lo stato DFA mostrano invece la destinazione della transizione (`→qN`).

I notebook attivi sono:

- `notebook/notebook_multilevel_framework.ipynb`;
- `notebook/notebook_manual_experiment.ipynb`.

Per riallinearne le celle ai sorgenti correnti:

```bash
python3 notebook/sync_training_notebooks.py
```

Lo script `experiment_comparison.py` rimane nella root per generare grafici e
confrontare i risultati dei nuovi esperimenti.

## Materiale accantonato

Le implementazioni multi-epsilon, DSAC/SAC, dual-learner, i relativi notebook
e tutti gli esperimenti preesistenti sono conservati senza eliminazioni in
`backup/legacy_2026-08-19/`. Il manifesto in quella directory descrive il
contenuto e indica come ripristinarlo.
