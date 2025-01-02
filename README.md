
# Deep-CBN: Integrating Convolutional Layers and Biformer Network with Forward-Forward and Backpropagation Training

## Abstract
<div align="justify">
Accurate molecular property prediction is crucial for drug discovery and computational chemistry, facilitating the identification of promising compounds and accelerating therapeutic development. Traditional machine learning falters with high-dimensional data and manual feature engineering, while existing deep learning approaches may not capture complex molecular structures, leaving a research gap. We introduce Deep-CBN, a novel framework designed to enhance molecular property prediction by capturing intricate molecular representations directly from raw data, thus improving accuracy and efficiency. Our methodology combines Convolutional Neural Networks (CNNs) with a BiFormer attention mechanism, employing both the Forward-Forward algorithm and backpropagation. The model operates in three stages: (1) Feature Learning, extracting local features from SMILES strings using CNNs; (2) Attention Refinement, capturing global context with a BiFormer module enhanced by the Forward-Forward algorithm; and (3) Prediction Subnetwork Tuning, fine-tuning via backpropagation. Evaluations on benchmark datasets—including Tox21, BBBP, SIDER, ClinTox, BACE, HIV, and MUV—show that Deep-CBN achieves near-perfect ROC-AUC scores, significantly outperforming state-of-the-art methods. These findings demonstrate its effectiveness in capturing complex molecular patterns, offering a robust tool to accelerate drug discovery processes.
</div>


## Method
<img width="610" alt="image" src="https://github.com/akianfar/Deep-CBN/blob/main/assets/Artboard%202-20.jpg">
<img width="610" alt="image" src="https://github.com/akianfar/Deep-CBN/blob/main/assets/Artboard%203-20.jpg">

### Requirements 

```
numpy==1.26.4
pandas==2.2.2
tensorflow==2.17.1
keras==3.5.0
seaborn==0.13.2
scikit-learn==1.6.0
einops==0.8.0

```
### Usage example

To use this model, navigate to the `src` folder and locate the corresponding `.ipynb` notebook file. In the cell containing the following line:

```python
data = pd.read_csv('/content/Deep-CBN/Data/tox21.csv')
```

You can modify the dataset path to match the dataset you intend to use. Additionally, update the label column reference accordingly based on the task of your selected dataset. For example:

```python
labels = data['NR-PPAR-gamma']
```

Replace `'NR-PPAR-gamma'` with the appropriate label column name corresponding to your specific task. After making these changes, execute the code to run the model with your desired dataset and task configuration.
