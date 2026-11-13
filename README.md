#  AirNox : Bioaerosol Intelligence & Computer Vision Research Platform

Making microscopic allergen detection more accessible through computer vision.

AirNox is a research-driven computer vision prototype exploring the classification of microscopic pollen grains, mold spores, and non-biological particulate matter using deep learning and explainable AI.

While academic and industrial research in automated palynology already exists, AirNox represents my personal engineering take on the problem: an attempt to bridge software classification with practical, micro-environmental exposure monitoring.

## Why I Started AirNox

As someone who personally suffers from pollen and other bioaerosol allergies, I have experienced how unpredictable pollen exposure can be and how significantly it can affect everyday life. This motivated me to explore whether technology could make environmental allergen detection more accessible.

The idea initially developed through my BET300 course at the University of Waterloo, where I began explorin the possibility of building a portable pollen detection device.

Before committing to a technical direction, I conducted a customer discovery and validation process involving 50+ interviews with people across different backgrounds, including researchers, professors, sports scientists, and parents of children who experience allergies.

These conversations helped me better understand the challenges people face when dealing with environmental allergens and the potential value of more accessible exposure-monitoring technology.

One particularly important insight came from discussions with professors: developing a portable air-sampling device would require careful consideration of airflow requirements, including an air intake target of approximately 18 L/min for the concept I was investigating.

This introduced a much larger engineering challenge involving airflow, particle collection, hardware integration, and microscopic identification.

Rather than attempting to solve every part of the problem simultaneously, I decided to focus on the software first.

AirNox is my first step toward that vision.

I am currently exploring whether computer vision and deep learning can reliably identify different microscopic particles from captured images. By developing and evaluating the classification pipeline independently, I hope to establish a foundation that could eventually integrate with a portable hardware platform.

My long-term ambition is to contribute to more accessible environmental allergen monitoring and, hopefully, help people living with allergic rhinitis better understand their exposure.

## Key Features
- Deep Learning Classification: Fine-tuned ResNet18 and EfficientNet backbones for classifying microscopic images across five categories: birch pollen, ragweed pollen, grass pollen, mold spores, and benign dust.
Explainable AI (Grad-CAM): Generates attention heatmaps to visualize image regions influencing model predictions.
- Image Quality Assessment: Uses Laplacian variance to evaluate optical sharpness and flag potentially unreliable microscope images.
Prediction Uncertainty: Provides confidence-based warnings to encourage manual verification of uncertain predictions.
- Scientific Reference Database: Connects predicted particle categories to biological morphology, seasonal patterns, and environmental exposure context.
- Interactive Streamlit Interface: Provides a research dashboard for image uploads, classification results, visual explainability, and scientific reference information.
- Apple Silicon Support: Designed to support Mac-native execution, with MPS acceleration where supported by the installed PyTorch environment.
Current Research Scope

AirNox currently focuses on the software component of a larger portable pollen detection concept.

The research prototype explores several questions:

- Can convolutional neural networks distinguish different pollen and fungal spore categories using microscopic images?
How does image sharpness affect classification reliability?
- Can Grad-CAM provide useful visual explanations of model predictions?
- How does model performance change across different microscopy sources, image conditions, and particle morphologies?
How could a future portable air-sampling device integrate with the image classification pipeline?

An important part of this work is understanding the limitations of computer vision in scientific applications. Model confidence does not necessarily represent prediction correctness, and visual attention maps do not independently establish biological validity.


## Technology Stack
Category	Technologies
Programming	Python
Deep Learning	PyTorch
Model Architectures	ResNet18, EfficientNet
Computer Vision	OpenCV, Pillow
Explainability	Grad-CAM
Interface	Streamlit
Numerical Computing	NumPy
Hardware Acceleration	Apple Silicon MPS
Current Status

AirNox is an ongoing independent research and development project.

The current focus is building and evaluating the microscopic image classification pipeline before moving toward hardware integration and portable air-sampling capabilities.

Future development directions include:

Improving classification performance and generalization.
Expanding and validating the microscopic image dataset.
Investigating domain shift across different microscopes and imaging conditions.
Exploring model calibration and more rigorous uncertainty estimation.
Investigating integration with a portable air-sampling and imaging system.
Limitations and Disclaimer

AirNox is an experimental research prototype, not a clinically validated environmental monitoring instrument or medical diagnostic device.

Classification outputs and allergen reference information are provided for research and environmental context only. Model predictions require independent validation, and the prototype should not be used to make medical decisions.

## Get Involved

AirNox started with a personal problem, conversations with people experiencing similar challenges, and a curiosity about whether engineering could help address it.

I am interested in connecting with researchers, professors, engineers, aerobiologists, environmental scientists, and anyone working on pollen monitoring, microscopy, bioaerosols, or accessible environmental sensing.

If you are interested in collaborating, sharing research insights, discussing the project, or exploring potential hardware integration, I would love to hear from you!

- Portfolio: https://prishitasingh06.github.io/portfolio/
- Email: p323sing@uwaterloo.ca

I hope that one day, projects like AirNox can contribute to making environmental allergen information more accessible and help people living with allergic rhinitis better understand the air around them!

## References & Related Research

1. **Sevillano, V., & Aznarte, J. L. (2018).** Improving classification of pollen grain images of the POLEN23E dataset through three different applications of deep learning convolutional neural networks. *PLOS ONE*, 13(8), e0201807. [https://doi.org/10.1371/journal.pone.0201807](https://doi.org/10.1371/journal.pone.0201807)
2. **Sevillano, V., Holt, K., & Aznarte, J. L. (2020).** Precise automatic classification of 46 different pollen types with deep learning. *PLOS ONE*, 15(6), e0234710. [https://pmc.ncbi.nlm.nih.gov/articles/PMC7310700/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7310700/)
3. **Cao, N., Meyer, M., Thiele, L., & Saukh, O. (2020).** Automated and continuous pollen detection with an affordable technology. *Proceedings of the European Conference on Smart Objects, Systems and Technologies (EWSN)*. Zenodo. [https://doi.org/10.5281/zenodo.3572653](https://doi.org/10.5281/zenodo.3572653)
4. **Struniawski, K., Machlanska, A., Marasek-Ciolakowska, A., & Konopka, A. (2026).** Automated pollen classification via subinstance recognition: A comprehensive comparison of classical and deep learning architectures. *MDPI Applied Sciences*, 16(2), 720. [https://doi.org/10.3390/2076-3417/16/2/720](https://doi.org/10.3390/2076-3417/16/2/720)
5. **Banach, J. K., et al. (2025).** Artificial intelligence in honey pollen analysis: Accuracy and limitations of pollen classification compared with palynological expert assessment. *MDPI Applied Sciences*, 15(23), 12848. [https://doi.org/10.3390/2076-3417/15/23/12848](https://doi.org/10.3390/2076-3417/15/23/12848)
6. **Gonçalves, A. B., et al. (2016).** Feature extraction and machine learning for the classification of Brazilian savannah pollen grains. *PLOS ONE*, 11(6), e0157044. [https://doi.org/10.1371/journal.pone.0157044](https://doi.org/10.1371/journal.pone.0157044)

## Project Structure
```text
AirNox/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── models/
│   └── checkpoints/
├── research/
│   ├── experimental_results.md
│   ├── literature_review.md
│   └── research_questions.md
├── src/
│   ├── data/
│   ├── explainability/
│   ├── models/
│   └── training/
├── requirements.txt
├── run_mac.sh
└── README.md
```

# Getting Started (macOS)

## Prerequisites

- **macOS:** Apple Silicon (M1/M2/M3/M4 recommended for MPS acceleration) or Intel Mac.
- **Python:** 3.9+
- **Git:** Installed.

## Option 1: Automated Setup (Recommended)

Clone the repository:

```bash
git clone https://github.com/prishitasingh06/AirNox.git
cd AirNox
```

Make the launcher executable and run it:

```bash
chmod +x run_mac.sh
./run_mac.sh
```

The script automatically:
- Creates a Python virtual environment.
- Installs dependencies.
- Downloads and preprocesses the dataset.
- Trains the ResNet18 baseline model.
- Launches the Streamlit dashboard.

Once complete, access the dashboard at [http://localhost:8501](http://localhost:8501).

## Option 2: Manual Setup

Run each pipeline stage individually.

**1. Create and activate a virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate
```

**2. Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**3. Download and preprocess the dataset**

```bash
python3 -m src.data.download_dataset
python3 -m src.data.preprocessing
```

**4. Train the classification model**

```bash
python3 -m src.training.train
```

**5. Launch the Streamlit dashboard**

```bash
streamlit run app/streamlit_app.py
```
