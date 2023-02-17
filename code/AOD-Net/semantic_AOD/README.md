Call interface: `run.sh` to fulfill train, validate, test, or dehaze.

e.g.

	bash run.sh train 0

---
### Code structure:
- `train.py` Train on ITS/OTS of RESIDE dataset, including validation.  
- `test.py` Test on SOTS of RESIDE dataset, evaluation metrics are PSNR and SSIM, dehazed images could be saved in folder `test_results`.
- `dehaze.py`  Dehazing demo, dehazed images saved in folder `demo_results`.
- `net.py` Network, modified baseline-AOD on the basis of original AOD-Net, introducing attention mechanism (PA, SA), etc. 
- `resnet.py` RefineNet.
- `dataloader.py` Preprocess data.
- `common.py` Hyperparameters (could be packaged to `args`), and normalization function. 
- `validate.py` Just validate.

---
### Tips
- The code could be further optimized. For example, the function `populate_*_list()` and `eval_index()` in `dataloader.py,train.py` and `test.py` could be merged and stored in the newly created `utils.py`. 
- Need to create folder `train_logs`  manually before training，could be optimized in code.
