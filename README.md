# DASC_2024
## MURDOC: Transforming Pixels into Perception for Camouflage Detection
The project in this repo got corrupted, therefore to accesss the working project go here: https://github.com/enjelika/MURDOC_2024

## How to Setup MURDOC Python Environment
### Required NuGet Packages
- ImageProcessor
- pythonnet

### Setup instructions
- Create an **user** environment variable named 'PythonDLL' and set the path to the python39 DLL you intend to use. (Such as one in a virtual environment).
- In **MainWindowViewModel.cs**, find the function: InitializePythonEngine and update the **pathToVirtialEnv** value to the location of your python environment. (The current version references a miniconda environment.)

## Research Questions
1. How XAI off-ramps enhance MURDOC's trustworthiness by providing transparent and reliable explanations of the decision-making process in camouflage detection scenarios?
2. To what extent does the integration of user-controllable image enhancement functionalities, such as brightness, contrast, and saturation adjustments, within the MURDOC visualization tool contribute to enhancing trust, usability, and understanding in camouflage detection?

## Abstract
The MURDOC project introduces an application to enhance trustworthiness and explainability in computer vision models, focusing on camouflage detection. It aims to address the need for transparent and interpretable AI systems in sensitive domains such as security and defense. MURDOC integrates advanced eXplainable AI techniques including off-ramps for collecting interpretable insights, attention mechanisms for highlighting relevant features, and a user-centric image pre-processing tool within the visualization interface. This paper offers an assessment of MURDOC's potential impact on trustworthiness and explainability in camouflaged object detection tasks. It discusses and assesses the potential effectiveness of off-ramps in gathering XAI output at various model stages, such as feature maps, attention mechanisms, and activation maps. Additionally, the paper investigates how user-driven image pre-processing mechanisms may enhance the trustworthiness of the model's predictions and decisions, allowing users to modify the input and observe prediction changes. As a work in progress, the development of MURDOC shows promise in bridging the gap for transparency and interpretability in camouflage detection. The paper also discusses the challenges, future directions, and potential applications of MURDOC's capabilities in diverse domains requiring transparent and trustworthy AI systems.
