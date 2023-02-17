Call interface: `run.sh` to fulfill train, validate, test, or dehaze.

e.g.

	bash run.sh train 0

---
### Code structure:
- `train.py` Train on ITS/OTS of RESIDE dataset, including validation.  
- `test.py` Test on SOTS of RESIDE dataset, evaluation metrics are PSNR and SSIM, dehazed images could be saved in folder `test_results`.
- `255_dehaze.py`  Dehazing demo, dehazed images saved in folder `demo_results`.
- `GCANet.py` Network, modified baseline-GCA on the basis of original GCA-Net, introducing attention mechanism (PA, SA), etc. 
- `resnet.py` RefineNet.
- `dataloader.py` Preprocess data.
- `common.py` Hyperparameters (could be packaged to `args`)
- `utils.py` Some useful function.
