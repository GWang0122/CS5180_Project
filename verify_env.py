"""Quick smoke test — run after activating the conda env to verify everything works."""

import sys

def check(name, import_fn):
    try:
        mod = import_fn()
        ver = getattr(mod, "__version__", "ok")
        print(f"  [PASS] {name:20s} {ver}")
        return True
    except Exception as e:
        print(f"  [FAIL] {name:20s} {e}")
        return False

print("=" * 50)
print(f"Python {sys.version}")
print("=" * 50)

all_ok = True
all_ok &= check("torch", lambda: __import__("torch"))
all_ok &= check("gymnasium", lambda: __import__("gymnasium"))
all_ok &= check("minigrid", lambda: __import__("minigrid"))
all_ok &= check("numpy", lambda: __import__("numpy"))
all_ok &= check("matplotlib", lambda: __import__("matplotlib"))
all_ok &= check("tensorboard", lambda: __import__("tensorboard"))

print("-" * 50)

import torch
if torch.cuda.is_available():
    print(f"  CUDA available: {torch.cuda.get_device_name(0)}")
else:
    print("  WARNING: CUDA not available — training will use CPU")

print("-" * 50)

import gymnasium as gym
import minigrid
gym.register_envs(minigrid)
for env_id in ["MiniGrid-MemoryS11-v0", "MiniGrid-DoorKey-8x8-v0"]:
    try:
        env = gym.make(env_id)
        obs, _ = env.reset()
        print(f"  [PASS] {env_id}  obs shape={obs['image'].shape}  actions={env.action_space.n}")
        env.close()
    except Exception as e:
        print(f"  [FAIL] {env_id}  {e}")
        all_ok = False

print("=" * 50)
print("ALL GOOD!" if all_ok else "Some checks failed — see above.")
