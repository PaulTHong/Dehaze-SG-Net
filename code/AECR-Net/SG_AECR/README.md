Call interface: `run.sh` to fulfill train, validate, test, or dehaze.

e.g.

	bash run.sh train 0

---
### Code structure:
- `train.py` Train with or without semantic guidance.
- `test.py` Test metrics are PSNR and SSIM, dehazed images could be saved in folder `test_results`.
- `AECRNet.py` Networks. 
- `resnet.py` RefineNet.
- `CR.py` Contrastive loss.
- `dataloader.py` Preprocess data.
- `utils/*` Some useful function.
