
Call interface: `run.sh` to fulfill train, test, or dehaze.

e.g.

	bash run.sh train 0

---
### Code structure:
- `SG_main.py` Train SG-FFA on ITS/OTS of RESIDE dataset, including test on SOTS.  
- `SG_test.py` Test SG-FFA on SOTS of RESIDE dataset, evaluation metrics are PSNR and SSIM, dehazed images could be saved in folder `test_results`.
- `SG_dehaze.py`  SG-FFA Dehazing demo, dehazed images saved in folder `demo_results`.
- `data_utils.py` Preprocess data.
- `models/` Networks:
    - `seg_att_FFA.py` SG-FFA.
    - `FFA.py` Original FFA-Net.
    - `resnet.py` RefineNet for semantic segmentation.
- `option.py`

---
- `main.py` Train of original FFA-Net.
- `test.py`  Test of original FFA-Net.
- `dehaze.py` Dehazing demo of original FFA-Net.
- `metrics.py` Calculate PSNR and SSIM.
- `utils.py` Some useful function.

---
### Tips
- Released models contain `.module`, so you could parallel your net with `nn.DataParallel()` first before loading the model.
