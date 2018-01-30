# Ranker

## File Description:

- `data/` : folder containing the data extracted from the database and the first human round evaluation,
with `[...].json` as json files containing the raw data, and `[...].features/` as folders containing json files for each hand-crafted feature.
- `embedding_metrics.py` : util file to compute embedding matching score between two sentences: greedy embedding match, extrema embedding match, average embedding match.
- `estimators.py` : tensorflow implementation of fully-connected feed-forward Neural Network to predict a candidate response thumbs -up/-down (SHORT-TERM mode) or the dialog final score (LONG-TERM mode).
- `explore.sh` : script to train a bunch of estimators and do parameter search
- `extract_dialogues_from_db.py` : extract (article, context, candidate) triples from the database from inner-lab data collection phase.
- `extract_dialogues_from_round1.py` : extract (article, context, candidate) triples from the first human-evaluation round of the ConvAI competition.
- `extract_transitions_from_db.py` : extract (s, a, r, vec) tuples from the AMT collected database,
With `s` the state composed of (article, context), `a` the action composed of (candidate), `r` the reward (either 0 or 1) and `vec` the custom encoding of (article, context, candidate).
- `features.py` : util file to compute hand-engineered features for (article, context, candidate) triples.
- `models/` : folder containing trained models. Each sub directory is for the type of model and its validation accuracy.
- `q_networks.py` : pytorch implementation of deep and very deep Q-neural networks to predict a q-value for each candidate utterance at any given timestep.
- `q_train.py` : _TODO_ implementation of pytorch training loop for q-networks.
- `test.py` : implementation of tensorflow test loop for estimators.
- `test.sh` : run script example of `test.py`.
- `train.py` : implementation of tensorflow training loop for estimators.
- `train.sh` : run script example of `train.py`.

