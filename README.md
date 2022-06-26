# MuZeroGoJax

Mu Zero Go implemented with [GoJAX](https://github.com/aigagror/GoJAX).

## Update step

![update step diagram](images/update_step.png)

## Training configs

#### Simple linear model

* black perspective embedding
* real transition
* linear value and policy

```shell
--batch_size=2 --board_size=7 --max_num_steps=98 --learning_rate=0.01 \
--training_steps=1 --eval_frequency=0 \
--embed_model=black_perspective --value_model=linear --policy_model=linear \
--transition_model=black_perspective
```

#### Lite CNN model

* 32 hidden dimensions
* black perspective + 1-layer CNN embedding
* 1-layer CNN transition
* linear value and policy

```shell
--batch_size=2 --board_size=7 --max_num_steps=98 --learning_rate=0.01 \
--training_steps=1 --eval_frequency=0 \
--embed_model=black_cnn_lite --value_model=linear --policy_model=cnn_lite \
--transition_model=cnn_lite
```

#### Intermediate CNN model

* 256 hidden dimensions
* black perspective + 4-layer CNN embedding
* 1-layer CNN transition
* linear value and policy

```shell
--batch_size=2 --board_size=7 --max_num_steps=98 --learning_rate=0.01 \
--training_steps=1 --eval_frequency=0 \
--embed_model=black_cnn_intermediate --value_model=linear --policy_model=cnn_lite \
--transition_model=cnn_intermediate
```
