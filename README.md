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

Training e valutazione verificano l'appartenenza sullo stato continuo. Per il
planning astratto, ogni regione etichetta tutte le celle che interseca; ogni
livello della gerarchia viene rasterizzato direttamente dalla regione continua.

Nel `multilevel_framework`, tutti i livelli non-top usano automaticamente il
dual Q-learning. Sul livello top si può scegliere tra VI e Q-learning classico
senza shaping. Gli iperparametri si configurano nel file di astrazione:

```json
{
  "name": "level1",
  "grid_w": 12,
  "grid_h": 12,
  "algorithm": "learning",
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
livello inferiore. Il top usa VI per default; impostando `"algorithm":
"learning"` usa invece una sola Q-table con la reward originale. Questo vale
anche per una gerarchia con un solo livello. Se la sezione `learning` manca
vengono usati i valori predefiniti mostrati sopra.
Negli stati DFA accettanti l'unica azione disponibile è `done`: assegna il
`goal_reward` e termina senza bootstrap. Il valore del goal viene quindi
appreso o calcolato dalla VI, senza inizializzare manualmente le Q-table.

Per ogni livello appreso vengono salvati `reward_epsilon.png` e i relativi dati
in `reward_epsilon_data.npz`, sotto
`results/<esperimento>/img/abstract_learning/levelN/`. Il grafico mostra gli
andamenti delle reward biased e unbiased insieme a epsilon.
Per ogni livello, indipendentemente dall'algoritmo VI o learning, la V-function
numerica viene inoltre salvata in
`results/<esperimento>/results/abstract_value_functions/levelN/value_function.npz`.
Le chiavi `unbiased_values` e, per i livelli dual, `biased_values` usano
l'ordinamento `[q_index, y, x]`; `values` rimane un alias dell'unbiased per
compatibilita. `dfa_states` associa ogni `q_index` all'identificatore reale
dello stato DFA; il file include anche dimensioni, gamma, goal reward,
algoritmo risolutivo e il flag `has_biased_values`.
Questo salvataggio viene prodotto anche con `--no-heatmaps`.
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
reward–epsilon e i log astratti vengono comunque prodotti.

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
