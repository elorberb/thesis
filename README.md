# 🍀 Automated Phenotyping of Cannabis Flower Maturity: A Machine Learning Approach

This research introduces a novel method combining machine learning and computer vision to accurately determine the maturity of cannabis flowers. This is achieved by analyzing trichomes, small hair-like structures on the flowers that are rich in cannabinoids, key ingredients in medical cannabis. The method focuses on the density and appearance of trichomes to gauge flower maturity, which is vital for optimal THC production. The goal is to refine harvest timing, enhance product quality, ensure stable medicinal properties, and reduce cultivation risks. The hypothesis suggests that this approach will greatly improve the efficiency and accuracy of maturity assessment, leading to more consistent and high-quality cannabis products. This research could significantly impact and potentially transform current practices in cannabis cultivation.

<p align="center">
  <img src="readme_images/cannabis flowers.jpg" alt="Cannabis Flower" width="400" height="400">
</p>

## 📑 Table of Contents
1. [🎯 Project Overview and Motivation](#project-overview-and-motivation)
2. [🔍 Why Cannabis Flower Maturity Matters](#why-cannabis-flower-maturity-matters)
3. [🚫 The Challenge](#the-challenge)
4. [🛠️ How We Approach It](#how-we-approach-it)
5. [📊 Dataset](#dataset)
6. [🏷️ Labeling Guidelines](#labeling-guidelines)

## 🎯 Project Overview and Motivation
### Why Cannabis Flower Maturity Matters
Cannabis flower maturity is a critical factor in determining the quality and potency of the final product. As cannabis flowers mature, their trichomes undergo a notable color transformation, shifting from clear to cloudy, and finally to amber. This color change is a vital indicator of the flower's maturity stage and directly impacts the chemical composition and potential effects of the final cannabis product. By employing computer vision techniques, we aim to segment the trichomes precisely assess these color changes to determine the optimal time for harvesting. Accurately gauging the maturity stage through these color indicators can significantly enhance the quality control process in cannabis cultivation, leading to a product that meets specific standards for medical or recreational use. 


<p align="center">
  <img src="readme_images/trichome color change cut.png" alt="Cannabis Flower">
</p>

### 🚫 The Challenge

The current standard for assessing the maturity of cannabis flowers involves the use of a loupe(magnifier). This manual method requires observers to closely inspect the trichomes on the cannabis flower to determine their color and clarity, which indicate the flower's maturity stage. However, this approach is highly subjective and can lead to significant variation in assessments between different observers.

<p align="center">
  <img src="readme_images/current measurment aproach using loupe.jpg" alt="Manual Inspection Using a Loupe"  width="300" height="300">
</p>

In contrast, our project employs advanced image analysis techniques to standardize and automate this process. By using mobile phone camera with a macro lens and computer vision algorithms to detect the color and clarity of trichomes, our method provides a more objective and deterministic approach to assessing maturity. This not only reduces the variability in assessments but also increases the accuracy and reliability of the maturity determination.

<p align="center">
  <img src="readme_images/our approach for the measurment using phone.jpg" alt="Manual Inspection Using a Loupe"  width="400" height="400">
</p>


### 🛠️ How We Approach It
We utilize Mask R-CNN, a powerful image segmentation model, to accurately identify and analyze trichomes in high-resolution images. This approach offers a more objective and consistent method for assessing flower maturity.


## 📊 Dataset
The dataset was created in partnership with Rohama Greenhouse and spans 5 latest weeks of cannabis flower growth. Using an iPhone 14 Pro with a 10X magnifying lens, the images provide a detailed view of the trichomes, essential for accurate analysis.


<p align="center">
  <img src="readme_images/images collection process.png" alt="Collecting data process">
</p>

## 🏷️ Labeling Guidelines
Labeling is performed through the segments.ai interface, allowing for accurate and consistent annotation of trichomes. This step is vital for training the Mask R-CNN model effectively.
