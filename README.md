# DASC_2024
## MURDOC: Transforming Pixels into Perception for Camouflage Detection

## Research Questions
1. How does incorporating off-ramps for eXplainable AI (XAI) output enhance the trustworthiness of the MURDOC application by providing transparent and reliable explanations of the decision-making process?
2. To what extent does the incorporation of user-driven verification mechanisms within the MURDOC visualization tool improve the trustworthiness of the model's predictions in camouflage detection, allowing users to actively participate in decision-making processes and validate model outputs based on their knowledge and experience? 
3. How can the integration of advanced model validation techniques, such as advesarial testing and uncertainty quanitification, within the MURDOC framework enhance the model's robustness and reliability in challenging camouflage scenarios, consequently improving the user trust in the model's performance and decision-making capabilities?

## Work Completed
2024-03-25: Completed the "off-ramps" for XAI output of ResNet50 and RankNet models

2024-04-01: Visual Studio 2022 C# .NET project created

## TODOS
1. Add "off-ramps" for XAI output of EfficientDet-D7 model
2. Complete design utilizing Gestalt principles and xaml coding of MURDOC GUI 
3. Wire up C# front-end with Python back-end (utilize pythonnet lib)
4. Incorporate cybersecurity elements to project

### Abstract
This paper introduces the Mixed Understanding and Recognition for Detection of Camouflage (MURDOC) project, which aims to enhance explainability in computer vision models, specifically focusing on camouflage detection. The project seeks to develop tools and methodologies to clarify decision-making processes and improve user understanding of camouflaged object detection and segmentation (CODS). Examining the critical necessity for explainability in computer vision, particularly in comprehending the decision-making procedures of machine learning (ML) and artificial intelligence (AI) models, forms the foundation of this research.

Camouflage detection presents intricate challenges to computer vision approaches due to the complex and adaptive techniques that can influence detection down to the pixel level. MURDOC develops an offline interactive visualization tool designed with a human-in-theloop
approach. This approach enables a more direct and comprehensive application of understanding of decision-making processes than in existing models. Through the integration of insights from Informative Artificial Intelligence (IAI) and eXplainable AI (XAI), MURDOC
aims to shed light on the decision-making processes inherent in contemporary computer vision models, including ResNet50, RankNet, and EfficientDet-D7.

Key initiatives within the MURDOC project include identifying and defining significant decision levels within prominent camouflage detection models. These decision levels are intended to be presented in the XAI output using various options, such as image depictions,
text-based outputs, and color-coded indicators, ensuring a detailed and nuanced understanding of the decision-making hierarchy within these models.

MURDOC pursues these initiatives following a human-centric design philosophy. The project updates the Find and Acquire Camouflage Explainability (FACE) model to implement these initiatives and accommodate changes in decision-level representation. An application
tool offers a learnable and usable graphic user interface that displays the original image alongside segmentation, IAI, and XAI outputs. This application aims to facilitate user viewing and interaction with model outputs to help them better understand the models and their predictions.

MURDOC prioritizes principles such as transparency, trust, robustness, and interpretability, recognizing the importance of explainability in AI. Transparency ensures stakeholders grasp the decision-making process, fostering understanding and accountability. Trust involves
assessing the confidence level of human users when interacting with the AI system and acknowledging the importance of building trust by enhancing transparency and interpretability.

Ensuring the robustness of MURDOC involves maintaining the resilience of models against shifts in data or parameters, providing consistent and dependable performance across diverse situations. The project adopts a multidisciplinary approach integrating transparency, trust, and cybersecurity, emphasizing reliability and security in camouflage detection. Robust privacy measures, including encryption and access controls, are incorporated into MURDOC’s design to protect future data. The commitment to privacy principles establishes a foundation for responsible data practices, even in the absence of sensitive data in the current project scope. Security considerations such as defense-in-depth and minimized attack surfaces are also integral to MURDOC’s design. By integrating insights from psychology, the project seeks to align user-centric design with technical standards, ensuring an approach that addresses users’
cognitive and affective needs for an effective and trustworthy solution.

While privacy is a paramount concern in AI applications, the current iteration of MURDOC does not involve handling private or sensitive user information. The framework incorporates privacy safeguards, anticipating future developments involving such data. MURDOC’s commitment
to privacy principles establishes a foundation for responsible data practices, even if sensitive data is not currently part of the project scope.

MURDOC aims to make a substantial impact by going beyond tool functionality. The goal is to transform how we understand camouflage detection. As a model focused on transparency, ethics, and user-friendly design, MURDOC has the potential to shape advanced AI technologies. The commitment to transparency and user-centric principles positions MURDOC as a transformative model, fostering trust and comprehension. MURDOC strives to create an easy-to-use visualization tool that allows users to understand why models make certain predictions, ultimately enhancing trust in these predictions.
