---

<div align="center">
  <img src="https://github.com/Gaurav14cs17/YOLOX_pytorch/blob/main/data/1_Igh61mg6Qi6te_YJaQvO5w.png" width="350">
</div>

# YOLOX: You Only Look Once X

YOLOX is an **anchor-free version of YOLO**, designed for high-performance object detection. It simplifies the detection pipeline by removing the need for anchor boxes, making it more efficient and easier to train.

**Paper**: [YOLOX: Exceeding YOLO Series in 2021](https://arxiv.org/abs/2107.08430)

---

## Key Features
- **Anchor-Free**: Eliminates the need for anchor boxes, simplifying the model architecture.
- **High Performance**: Achieves state-of-the-art results in object detection tasks.
- **Flexible**: Supports various backbones and can be easily adapted to different use cases.

---

## Image Output Examples

<div align="center">
  <img src="https://github.com/Gaurav14cs17/YOLOX_pytorch/blob/main/data/nano_img_out/000026.jpg" width="400">
  <img src="https://github.com/Gaurav14cs17/YOLOX_pytorch/blob/main/data/nano_img_out/000054.jpg" width="400">
</div>

---

## Getting Started

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Gaurav14cs17/YOLOX_pytorch.git
   cd YOLOX_pytorch
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Training the Model

To train the YOLOX model, run the following command:
```bash
python train_nano_yolox.py
```

---

## Running the Demo

To perform object detection using the trained model, run the demo script:
```bash
python demo/nano_predict_model.py
```

---

## References
This implementation is based on the following resources:
1. [Official YOLOX Repository](https://github.com/Megvii-BaseDetection/YOLOX)
2. [PaddlePaddle PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)

---

## License
This project is open-source and available under the MIT License. For more details, see the [LICENSE](LICENSE) file.

---

## Contributing
Contributions are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.

---

## Acknowledgments
- Thanks to the authors of the YOLOX paper for their groundbreaking work.
- Special thanks to the contributors of the [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) repository for their implementation.

---
