# isaac_nav — 3D (Isaac Lab) training, rewritten from scratch

The same task as the 2D env (`dqn_2d_standalone_oscillation`), flown by the Isaac Lab multirotor at a fixed height.
It reuses the 2D code for the parts that must stay identical: obstacle layouts, the state vector, the reward and
the episode outcomes. Isaac only replaces the motion (physics plus a cascaded velocity controller) and the
distance readings (one forward OAK-D-Lite-like depth camera, 73° HFOV, binned into the same sectors as 2D).

| file | role |
|---|---|
| `nav_env.py` | Isaac `DirectRLEnv` and a wrapper with the 2D env's interface |
| `depth_bins.py` | depth image → per-sector minimum distance (pure torch) |
| `dqn_trainer.py` | the DQN loop (same algorithm as `train_dqn.py`) |
| `train_isaac.py` | training entry point (resume from a 2D checkpoint to fine-tune) |
| `eval_isaac.py` / `eval_core.py` | greedy evaluation; outcome rates, crash kinds, gap groups, camera-vs-geometry reading error |
| `step_response.py` / `response_fit.py` | Phase 0: measure the drone's lag, acceleration and delay |

The settings must match the 2D policy: `--camera_count 1 --sensor_min_range 0.4 --obs_velocity 1`.
`--sensor geometry` feeds the policy exact 2D readings at the Isaac pose; the gap between the camera run and
the geometry run is what the camera costs, and the gap between geometry and 2D is what the dynamics cost.

## Run order

```bash
# 0. step response: are tau / accel / delay inside the 2D training ranges?
python isaac_nav/step_response.py --headless

# 1. zero-shot evaluation of the 2D policy, camera and geometry
CK=dqn_2d_standalone_oscillation/runs/C2_cam1_min04/s8/checkpoints/s8_final.pt
COMMON="--camera_count 1 --sensor_min_range 0.4 --obs_velocity 1 --headless"
python isaac_nav/eval_isaac.py --checkpoint $CK --sensor camera   $COMMON
python isaac_nav/eval_isaac.py --checkpoint $CK --sensor geometry $COMMON

# 2. fine-tune in Isaac from the 2D policy
python isaac_nav/train_isaac.py --resume_from $CK --output_dir runs_isaac/ft1 --num_envs 64 $COMMON
```
