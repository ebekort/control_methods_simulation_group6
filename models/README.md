# Trained models

One model per development stage, each the seed with the lowest one-step test loss of its stage (so the choice uses no ground truth). All use `--actuation network`, 30 epochs, Adam with cosine decay, batch 256.

| File | Data (`pd+gravity_control_data.py`) | Training | Seed of |
|---|---|---|---|
| `best.pkl` | 1 kHz, `--controller pd --dither 1.0` (no model) | no `--a-reg` (final recipe) | 4 of 5 |
| `stage3_1khz_pd_areg.pkl` | 1 kHz, `--controller pd --dither 1.0` | `--a-reg 0.01` | 4 of 5 |
| `stage2_1khz_ct_areg.pkl` | 1 kHz, `--controller model --dither 1.0` (computed torque, exact M and G) | `--a-reg 0.01` | 1 of 5 |
| `stage1_100hz_ct_areg.pkl` | 100 Hz (`--dt 0.01`), `--controller model --dither 1.0` | `--a-reg 0.01` | 0 of 2 |
| `inject_1khz_ct.pkl` | 1 kHz, `--controller model --dither 1.0 --inject-a --inject-d` | no `--a-reg` | 0 of 3 |

`inject_1khz_ct.pkl` was trained on data whose input is u = A⁻¹τ with a known A ≠ I and an extra damping D; it is only meaningful with `pinn_eval.py` on such data. The closed-loop scripts drive PyBullet with u as the joint torque, so use the other models there, for example

    python scripts/pinn_control.py models/best.pkl --gui
