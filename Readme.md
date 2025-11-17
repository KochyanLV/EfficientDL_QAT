# Our project in quantization

## Clone project
```
git clone https://github.com/KochyanLV/EfficientDL_QAT.git
```

## Presentation

```
[LINK](https://github.com/KochyanLV/EfficientDL_QAT/blob/main/%D0%90%D0%BD%D0%B0%D0%BB%D0%B8%D0%B7%20%D0%BC%D0%B5%D1%82%D0%BE%D0%B4%D0%BE%D0%B2%20%D0%BA%D0%B2%D0%B0%D0%BD%D1%82%D0%B8%D0%B7%D0%B0%D1%86%D0%B8%D0%B8.pdf)
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

### Results for lstm

| Methods         | Roc-Auc |
|----------------|---------|
| BaseModel fp32 | 0.948   |
| LSQ            | 0.834   |
| PACT           | 0.944   |
| AdaRound       | 0.939   |
| APoT           | 0.943   |
| DoreFa (best)  | 0.947   |


## sasrec training from scratch
```
1. python sasrec/training.py - train models with QAT for sasrec
2. sasrec/sasrec-qat.ipynb - notebook with all experiments 
3. sasrec/outputs/ - all best checkpoints
```

### Results for sasrec

| Methods         | NDCG  |
|----------------|-------|
| BaseModel fp32 | 0.365 |
| LSQ            | 0.308 |
| PACT           | 0.221 |
| AdaRound       | 0.314 |
| APoT (best)    | 0.337 |
| DoreFa         | 0.300 |


## espcn training from scratch
```
./run_espcn_experiments.sh
```

### Results for espcn

| Methods         | PCNR |
|----------------|------|
| BaseModel fp32 | 25.4 |
| LSQ            | 12.0 |
| PACT           | 25.2 |
| AdaRound       | 15.4 |
| APoT           | 7.4  |
| DoreFa (best)  | 13.3 |
| STE            | 24.9 |

