# Speed benchmark on best models

## To launch
```
python lstm.speed_test_best_model.py
```

| Model              | Time (s) | Val Loss | ROC AUC |   F1   | Precision | Recall |
|--------------------|---------:|---------:|--------:|:------:|:---------:|:------:|
| Base FP32          | 307.64   | 0.3048   | 0.9497  | 0.8760 | 0.8863    | 0.8659 |
| Base INT8          | 354.54   | 0.3045   | 0.9497  | 0.8765 | 0.8873    | 0.8658 |
| DoReFa QAT    | 122.34   | 0.3196   | 0.9463  | 0.8712 | 0.8926    | 0.8508 |
| DoReFa QAT (INT8)  | 190.17   | 0.3207   | 0.9462  | 0.8709 | 0.8921    | 0.8506 |

