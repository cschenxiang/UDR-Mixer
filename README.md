## 📖 Towards Ultra-High-Definition Image Deraining: A Benchmark and An Efficient Method
> Hongming Chen, Xiang Chen, Chen Wu, Zhuoran Zheng, Jinshan Pan, and Xianping Fu <br>

---
## 🔑 Setup
Type the command:
```
pip install -r requirements.txt
```

### 4K-Rain13k Dataset
![Example](figures/overview.png)
(The datasets are hosted on both Google Drive and BaiduPan)
| Download Link | Description | 
|:-----: |:-----: |
| Google Drive / [Baidu Netdisk](https://pan.baidu.com/s/1Kao-OjWNlgg2Jl0Jtl7e5Q?pwd=spfi) | A total of 12,500 pairs for training and 500 pairs for testing. |


## 🛠️ Training and Testing
1. Please download the corresponding datasets and put them in the folder `data/`.
2. Follow the instructions below to begin training our model.
```
python train.py
```
3. Follow the instructions below to begin testing our model.
```
python test.py
```
Run the script then you can find the output visual results in the folder `output/`.


### Evaluation
The PSNR, SSIM and MSE results are computed by using this [Python Code](https://github.com/cschenxiang/UDR-Mixer/tree/main/metrics).


### Visual Results
| Method | Download Link | 
|:-----: |:-----: |
| LPNet | Google Drive / [Baidu Netdisk](https://pan.baidu.com/s/1duS3geN2mEbWA3e2lYRJ_Q?pwd=br8d) |
| JORDER-E | Google Drive / [Baidu Netdisk](https://pan.baidu.com/s/1tMWkW8pGomOZvWV-ozDeBg?pwd=zghk) |
| RCDNet | Google Drive / [Baidu Netdisk](https://pan.baidu.com/s/1JPM9IjUonVJegQLET9-vxw?pwd=95s1) |
| SPDNet | Google Drive / [Baidu Netdisk](https://pan.baidu.com/s/1k0Vr_qX42JL_YxlMIVfurQ?pwd=r036) |
| IDT | Google Drive / [Baidu Netdisk](https://pan.baidu.com/s/1I7hBWfuozbt1m0LYRKi2qQ?pwd=bs9k) |
| Restormer | Google Drive / [Baidu Netdisk](https://pan.baidu.com/s/1MnagUIktnWzEOA20gyLslA?pwd=u77v) |
| DRSformer | Google Drive / [Baidu Netdisk](https://pan.baidu.com/s/1wCugfQmsGdojtUiZ5SABcQ?pwd=qumu) |
| UDR-S2Former | Google Drive / [Baidu Netdisk](https://pan.baidu.com/s/1OVbUMHgmEHAMt_0fo9JGZg?pwd=i4w5) |
| UDR-Mixer | Google Drive / [Baidu Netdisk](https://pan.baidu.com/s/1mo9tKs4FyDIaFyo9IGXThA?pwd=ghqi) |


### Citation
If you find this project useful in your research, please consider citing:
```
@article{chen2024towards,
  title={Towards Ultra-High-Definition Image Deraining: A Benchmark and An Efficient Method},
  author={Chen, Hongming and Chen, Xiang and Wu, Chen and Zheng, Zhuoran and Pan, Jinshan and Fu, Xianping},
  journal={arXiv preprint arXiv:2405.17074},
  year={2024}
}
```

### Disclaimer
Please only use the dataset for research purposes.

### Contact
If you have any questions, please feel free to reach me out at chenxiang@njust.edu.cn
