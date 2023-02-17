## Dehaze-SG-Net
 
- <font size=5>Official codebase of paper: **[SG-Net: Semantic Guided Network for Image Dehazing](https://openaccess.thecvf.com/content/ACCV2022/html/Hong_SG-Net_Semantic_Guided_Network_for_Image_Dehazing_ACCV_2022_paper.html)**, Tao Hong *et al.*, ACCV 2022.</font>

### Dataset
- `RESIDE`
- `NH-HAZE`
- `Dense-Haze`

### Code  
- Contain 5 folders: `AOD-Net`, `GCA-Net`, `FFA-Net`, `AECR-Net` and `semantic_segmentation`. 
- The former 4 correspond to our SG-AOD, SG-GCA, SG-FFA and SG-AECR. In every subfolder, there is a shell `run.sh`, get started from it.
- `semantic_segmentation` stores the called semantic segmentation model: `light-weight-refinenet`.

### Contact
- If you have any question, please contact with paul.ht@pku.edu.cn.