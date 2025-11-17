
# LSTM - Long-Short Term Memory with QAT

## Experiment Results

| epoch | train_loss | val_loss | roc_auc | f1    | precision | recall | json_name |
|-------|------------|----------|---------|-------|-----------|--------|-----------|
| 10 | 0.045 | 0.512 | ```0.948``` | 0.871 | 0.857 | 0.885 | ```baselstm```_vocab_size5000_emb_dim256_hidden_dim512_num_classes1_epochs10_bs64_lr0p0003.json |
| 5  | 0.212 | 0.313 | ```0.947``` | 0.872 | 0.891 | 0.854 | ```quantlstmdorefa```_vocab_size5000_emb_dim128_hidden_dim256_num_classes1_bits_a8_bits_w8_act_signedTrue_act_preproctanh_epochs5_bs128_lr0p001.json |
| 10 | 0.267 | 0.374 | ```0.944``` | 0.835 | 0.920 | 0.765 | ```quantlstmpact```_vocab_size5000_emb_dim256_hidden_dim512_num_classes1_bits_act8_bits_w8_pact_init_alpha6p0_epochs10_bs64_lr0p0003.json |
| 5  | 0.216 | 0.339 | ```0.943``` | 0.875 | 0.825 | 0.931 | ```quantlstmapot```_vocab_size5000_emb_dim128_hidden_dim256_num_classes1_bits8_k1_init_alpha_act6p0_init_alpha_w2p0_use_weight_norm_wTrue_epochs5_bs128_lr0p001.json |
| 10 | 0.065 | 0.443 | 0.944 | 0.872 | 0.872 | 0.872 | quantlstmapot_vocab_size5000_emb_dim256_hidden_dim512_num_classes1_bits8_k1_init_alpha_act6p0_init_alpha_w2p0_use_weight_norm_wTrue_epochs10_bs64_lr0p0003.json |
| 10 | 0.183 | 0.338 | 0.943 | 0.873 | 0.830 | 0.921 | baselstm_vocab_size5000_emb_dim128_hidden_dim256_num_classes1_epochs10_bs64_lr0p0003.json |
| 10 | 0.198 | 0.337 | 0.941 | 0.872 | 0.859 | 0.885 | quantlstmdorefa_vocab_size5000_emb_dim128_hidden_dim256_num_classes1_bits_a8_bits_w8_act_signedTrue_act_preproctanh_epochs10_bs64_lr0p0003.json |
| 10 | 0.167 | 0.365 | 0.939 | 0.857 | 0.888 | 0.828 | quantlstmadaround_vocab_size5000_emb_dim256_hidden_dim512_num_classes1_bits_w8_epochs10_bs64_lr0p0003.json |
| 10 | 0.187 | 0.356 | 0.939 | 0.870 | 0.847 | 0.895 | quantlstmapot_vocab_size5000_emb_dim128_hidden_dim256_num_classes1_bits8_k1_init_alpha_act6p0_init_alpha_w2p0_use_weight_norm_wTrue_epochs10_bs64_lr0p0003.json |
| 5  | 0.354 | 0.350 | 0.932 | 0.859 | 0.815 | 0.909 | quantlstmpact_vocab_size5000_emb_dim128_hidden_dim256_num_classes1_bits_act8_bits_w8_pact_init_alpha6p0_epochs5_bs128_lr0p001.json |
| 10 | 0.336 | 0.352 | 0.929 | 0.856 | 0.821 | 0.894 | quantlstmpact_vocab_size5000_emb_dim128_hidden_dim256_num_classes1_bits_act8_bits_w8_pact_init_alpha6p0_epochs10_bs64_lr0p0003.json |
| 5  | 0.317 | 0.350 | ```0.929``` | 0.855 | 0.812 | 0.903 | ```quantlstmadaround```_vocab_size5000_emb_dim128_hidden_dim256_num_classes1_bits_w8_epochs5_bs128_lr0p001.json |
| 10 | 0.276 | 0.384 | 0.928 | 0.859 | 0.822 | 0.899 | quantlstmadaround_vocab_size5000_emb_dim128_hidden_dim256_num_classes1_bits_w8_epochs10_bs64_lr0p0003.json |
| 10 | 0.568 | 0.544 | ```0.834``` | 0.776 | 0.723 | 0.836 | ```quantlstmlsq```_vocab_size5000_emb_dim256_hidden_dim512_num_classes1_bits8_epochs10_bs64_lr0p0003.json |
| 10 | 0.660 | 0.636 | 0.683 | 0.494 | 0.821 | 0.354 | quantlstmlsq_vocab_size5000_emb_dim128_hidden_dim256_num_classes1_bits8_epochs10_bs64_lr0p0003.json |
| 5  | 0.683 | 0.675 | 0.635 | 0.580 | 0.605 | 0.557 | quantlstmlsq_vocab_size5000_emb_dim128_hidden_dim256_num_classes1_bits8_epochs5_bs128_lr0p001.json |
