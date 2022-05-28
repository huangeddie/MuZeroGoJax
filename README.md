# MuZeroGoJax

Mu Zero Go implemented with [GoJAX](https://github.com/aigagror/GoJAX).

## Update step

![update step diagram](images/update_step.png)

## Simple black perspective, real, linear model training

```shell
--batch_size=2 --board_size=7 --max_num_steps=50 --learning_rate=0.01 \
--training_steps=5 --eval_frequency=0 \
--embed_model=black_perspective --value_model=linear --policy_model=linear \
--transition_model=black_perspective
```
