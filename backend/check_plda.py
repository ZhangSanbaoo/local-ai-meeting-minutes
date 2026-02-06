import numpy as np
from pathlib import Path

plda_dir = Path(r"C:\Project\local-ai-meeting-minutes\models\pyannote\plda-community-1\plda")

print("=== plda.npz ===")
d = np.load(plda_dir / "plda.npz")
for key in d.keys():
    print(f"  {key}: shape={d[key].shape}, dtype={d[key].dtype}")

print("\n=== xvec_transform.npz ===")
d2 = np.load(plda_dir / "xvec_transform.npz")
for key in d2.keys():
    print(f"  {key}: shape={d2[key].shape}, dtype={d2[key].dtype}")
