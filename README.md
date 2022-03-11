# Active Learning by Feature Mixing
PyTorch implementation of ALFA-Mix. For details, see Active Learning by Feature Mixing.

<div align="center">
  <img width="45%" alt="ALFA-Mix" src="Img1.png">
</div>

The code includes the implementations of all the baselines presented in the paper.

## Training
```python
python main.py --data_name dataset --data_dir your_data_directory --n_init_lb 100 --n_query 100 --n_round 15 --learning_rate 0.001 --n_epoch 1000 --model mlp --strategy AlphaMixSampling --alpha_opt --alpha_closed_form_approx --alpha_cap 0.2
```
