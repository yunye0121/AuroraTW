<!-- ```markdown -->
# AuroraTW

*A Taiwanese weather foundation model implementation.*

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Scripts](#scripts)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## Overview

**AuroraTW** is an open-source foundation model for weather forecasting and meteorological data analysis, tailored for the unique climate and terrain of Taiwan. It provides an end-to-end pipeline for training, inference, and evaluation, helping both researchers and practitioners accelerate their work in weather prediction.

---

## Features

- State-of-the-art model architecture for weather prediction in Taiwan
- Ready-to-use training and inference scripts
- Tools for preprocessing, reading, and handling NetCDF (`.nc`) weather data
- Modular Python code for easy extension and experimentation
- Shell scripts for batch automation

---

<!-- ## Directory Structure

```

aurora/                  # Core model and utility modules
ar\_gen\_eval\_Aurora.py    # AR generation & evaluation script for Aurora
dataset.py               # Dataset handling and loading utilities
inference.sh             # Shell script for running inference
read\_nc\_files.py         # Script to read NetCDF (.nc) weather data
single\_ar\_eval.py        # Evaluation script for single AR instances
single\_infer.sh          # Shell script for single inference runs
test\_dataset.py          # Script for dataset testing and validation
train.sh                 # Shell script for starting training
train\_Aurora.py          # Main training script for the Aurora model
utils.py                 # General utility functions

````

--- -->

## Our Contribution and Public Checkpoints:
We have released preliminary model checkpoints for non-commercial use.
While these models are publicly available and free to use,
please note that we do not guarantee 100% accuracy in the predictions or inferences made by the models.

Currently, we provide checkpoints for three different lead times: 1 hour, 3 hours, and 24 hours.
All checkpoints are saved in .pth format and will be made compatible with the new Safetensors format in the near future.

You can access all available checkpoints here:
https://drive.google.com/drive/folders/1PIS0pf3owXg69OA7TM_y79-djgTnGrcI

## Installation

### Requirements

- Python 3.8+
- numpy
- torch
- netCDF4
- tqdm
- (Add any other dependencies as needed)

### Setup

Clone this repository:
```bash
git clone https://github.com/yunye0121/AuroraTW.git
cd AuroraTW
````

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Training

To train the Aurora model:

```bash
bash train.sh
```

Or run directly:

```bash
python train_Aurora.py --config [your_config.yaml]
```

Replace `[your_config.yaml]` with your configuration file.

### Inference

To perform inference:

```bash
bash single_ar_eval.sh
```

<!-- Or run directly:

```bash
python ar_gen_eval_Aurora.py --input [input_file] --output [output_file]
``` -->

<!-- ### Dataset Preparation

Prepare your weather datasets in the required format. Use `read_nc_files.py` to preprocess NetCDF (`.nc`) files:

```bash
python read_nc_files.py --input [raw_data.nc] --output [processed_data.npy]
``` -->

<!-- ### Testing

To validate your dataset or model:

```bash
python test_dataset.py
``` -->

---

## Scripts

* **train\_Aurora.py** — Main model training script.
* **ar\_gen\_eval\_Aurora.py** — Script for AR (AutoRegressive) generation and evaluation.
* **dataset.py** — Dataset loading and preparation utilities.
* **test\_dataset.py** — Script for testing/validating datasets.
* **inference.sh / single\_infer.sh** — Shell scripts for batch or single inference runs.
* **single\_ar\_eval.py** — Evaluate single AR cases.
* **utils.py** — Helper and utility functions.

---

## Contributing

Contributions are welcome! Please open an issue to discuss your ideas, report bugs, or request features. Pull requests are encouraged.

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a pull request

---

## License

This project is licensed under the MIT License.
See the [LICENSE](LICENSE) file for details.

---

## Citation

If you use AuroraTW in your research or project, please cite us as follows:

```bibtex
@misc{auroratw2025,
  title   = {AuroraTW: A Taiwanese Weather Foundation Model Implementation},
  author  = {Yun-Ye, Cai},
  year    = {2025},
  url     = {https://github.com/yunye0121/AuroraTW},
  note    = {GitHub repository}
}
```

---

## Contact

For questions or collaborations, please open an issue or contact the repository maintainer.

---
