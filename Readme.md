# Our project in quantization

## Clone project
```
git clone https://github.com/KochyanLV/EfficientDL_QAT.git
```

## Install requirements.txt
```
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## lstm training from scratch
```
1. python lstm.lstm_utils.tokenize.py - builds tokenizer if you want to try another tokenization
2. python lstm.training.py - train models with qats
3. python lstm.speed_tets_best_model.py - small benchmark on best and base models
4. python lstm.vizualize_act_dorefa.py - vizualizing activation ranges and errors in dorefa
```

## sasrec training from scratch
```
1. python sasrec/training.py - train models with QAT for sasrec
2. sasrec/SASRec_QAT.pdf - notebook with all experiments 
3. sasrec/outputs/ - all best checkpoints
```
