# Relearning (only_forget data): Best, Average, and Worst Runs per SNMF Model

For each SNMF model, runs over all relearning learning rates are ranked by mean validation accuracy (average of the 8 arithmetic validation accuracies). Reported: **best**, **average** (run closest to the arithmetic mean of all runs), and **worst** run with their learning rate and metrics.

## gemma-2-0.1B_MaxEnt_lr_7.0e-05-partial_distill-alpha_0.1-beta_0.1-mask_snmf

Total runs (learning rates): 10

| Run type | Learning rate | Mean val acc | add_eq | add_wp | sub_eq | sub_wp | mult_eq | mult_wp | div_eq | div_wp | eng_ce |
|----------|----------------|--------------|--------|--------|--------|--------|---------|---------|--------|--------|-------|
| **Best** | 5e-05 | 0.7950 | 0.93 | 0.7 | 0.7 | 0.73 | 0.9 | 0.72 | 1.0 | 0.68 | 4.559327846364884 |
| **Average** (closest to mean) | 4e-06 | 0.7287 | 0.98 | 0.77 | 0.81 | 0.57 | 0.72 | 0.6 | 0.89 | 0.49 | 4.559670781893004 |
| **Worst** | 0.001 | 0.4538 | 0.02 | 0.0 | 0.01 | 0.01 | 0.96 | 0.78 | 0.96 | 0.89 | 14.253772290809328 |

## gemma-2-0.1B_MaxEnt_lr_7.0e-05-partial_distill-alpha_0.3-beta_0.1-mask_snmf

Total runs (learning rates): 10

| Run type | Learning rate | Mean val acc | add_eq | add_wp | sub_eq | sub_wp | mult_eq | mult_wp | div_eq | div_wp | eng_ce |
|----------|----------------|--------------|--------|--------|--------|--------|---------|---------|--------|--------|-------|
| **Best** | 7e-05 | 0.7975 | 0.84 | 0.7 | 0.74 | 0.64 | 1.0 | 0.77 | 1.0 | 0.69 | 4.592806927297668 |
| **Average** (closest to mean) | 4e-06 | 0.7200 | 0.97 | 0.71 | 0.82 | 0.53 | 0.73 | 0.6 | 0.88 | 0.52 | 4.565886488340192 |
| **Worst** | 0.001 | 0.4650 | 0.01 | 0.02 | 0.01 | 0.03 | 0.96 | 0.77 | 0.98 | 0.94 | 14.245884773662551 |

## gemma-2-0.1B_MaxEnt_lr_7.0e-05-partial_distill-alpha_0.6-beta_0.1-mask_snmf

Total runs (learning rates): 10

| Run type | Learning rate | Mean val acc | add_eq | add_wp | sub_eq | sub_wp | mult_eq | mult_wp | div_eq | div_wp | eng_ce |
|----------|----------------|--------------|--------|--------|--------|--------|---------|---------|--------|--------|-------|
| **Best** | 0.0001 | 0.7975 | 0.82 | 0.68 | 0.89 | 0.51 | 1.0 | 0.73 | 0.98 | 0.77 | 4.706575788751715 |
| **Average** (closest to mean) | 7e-06 | 0.6212 | 0.83 | 0.73 | 0.86 | 0.44 | 0.69 | 0.34 | 0.68 | 0.4 | 4.63931755829904 |
| **Worst** | 0.001 | 0.4562 | 0.02 | 0.0 | 0.01 | 0.05 | 0.95 | 0.77 | 0.97 | 0.88 | 15.202932098765432 |

## gemma-2-0.1B_MaxEnt_lr_7.0e-05-partial_distill-alpha_0.9-beta_0.1-mask_snmf

Total runs (learning rates): 10

| Run type | Learning rate | Mean val acc | add_eq | add_wp | sub_eq | sub_wp | mult_eq | mult_wp | div_eq | div_wp | eng_ce |
|----------|----------------|--------------|--------|--------|--------|--------|---------|---------|--------|--------|-------|
| **Best** | 7e-05 | 0.7888 | 0.85 | 0.69 | 0.72 | 0.65 | 0.97 | 0.74 | 1.0 | 0.69 | 4.67781207133059 |
| **Average** (closest to mean) | 7e-06 | 0.6300 | 0.87 | 0.6 | 0.69 | 0.44 | 0.73 | 0.47 | 0.75 | 0.49 | 4.642704046639232 |
| **Worst** | 0.001 | 0.4550 | 0.01 | 0.0 | 0.01 | 0.01 | 0.99 | 0.76 | 0.99 | 0.87 | 15.202932098765432 |

## gemma-2-0.1B_MaxEnt_lr_7.0e-05-partial_distill-alpha_1.0-beta_0.1-mask_snmf

Total runs (learning rates): 10

| Run type | Learning rate | Mean val acc | add_eq | add_wp | sub_eq | sub_wp | mult_eq | mult_wp | div_eq | div_wp | eng_ce |
|----------|----------------|--------------|--------|--------|--------|--------|---------|---------|--------|--------|-------|
| **Best** | 7e-05 | 0.7800 | 0.85 | 0.63 | 0.65 | 0.65 | 1.0 | 0.72 | 1.0 | 0.74 | 4.67772633744856 |
| **Average** (closest to mean) | 7e-06 | 0.6537 | 0.75 | 0.69 | 0.7 | 0.62 | 0.76 | 0.43 | 0.69 | 0.59 | 4.652349108367627 |
| **Worst** | 0.001 | 0.4437 | 0.02 | 0.0 | 0.02 | 0.01 | 0.96 | 0.74 | 0.96 | 0.84 | 14.103309327846365 |
