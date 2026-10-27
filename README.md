<picture>
  <source media="(prefers-color-scheme: dark)" srcset=".github/assets/banner-dark.png">
  <img alt="The University of Chicago, MS in Applied Data Science. ADSP 32014 course project." src=".github/assets/banner-light.png">
</picture>

# Industrial Biscuit Defect Detection

> **Example project from Spring 2025.** This is a copy of [mvemuri-22/Industry-Biscuit-Anomaly-Detection](https://github.com/mvemuri-22/Industry-Biscuit-Anomaly-Detection), shown with the authors' work and commit history intact. Their own write-up is [REPORT.md](REPORT.md). Course staff wrote this README and made the edits listed under [Changes by course staff](#changes-by-course-staff), including the section on [what to learn from this project](#what-to-learn-from-this-project).

On a biscuit production line, can a model that has only ever seen good biscuits flag the bad ones, and can it tell a harmless misshapen biscuit from one with a screw baked into it? The team trained four generative models on photographs of defect-free Tarallini biscuits and scored new biscuits by how badly each model reconstructs them.

## Summary

The data is a public Kaggle set of 1,225 biscuit photographs, each captured four times at 90-degree rotations, labeled defect-free or defective with three defect types: incomplete shape, foreign object, and color. The team split the data by groups of four rotations so the same biscuit never appears on both sides of a split, then trained four models on defect-free images only: a convolutional autoencoder as the non-Bayesian baseline, a variational autoencoder, a deep convolutional GAN scored by optimizing a latent code to reconstruct each test image, and a VAE whose encoder is a pretrained Vision Transformer. Reconstruction error above a threshold tuned on validation data marks a biscuit as defective. The GAN scored best in the authors' report, with precision and recall both around 0.85 to 0.92. K-means on the VAE's latent codes separated the incomplete-shape defects, which a bakery might still sell, from the foreign-object and color defects, which it must not. The Vision Transformer VAE reconstructed everything well, including the defects, and so failed at the task, which the authors explain correctly.

## Team and contributions

Team "Turing Tested", Spring 2025.

- **Alex Lee** ([@alexlee0422](https://github.com/alexlee0422)): the exploratory analysis, the Keras VAE, threshold tuning, the defect-type clustering, and much of the write-up.
- **Zimeng Huang** ([@Andy-hzm](https://github.com/Andy-hzm)): the Vision Transformer VAE and its analysis, and the write-up's equations.
- **Manas Vemuri** ([@mvemuri-22](https://github.com/mvemuri-22)): the repository, the data download and folder preparation, the convolutional autoencoder and the first VAE attempt.
- **Rithik Alluri**: the DCGAN notebook, committed to the repository by a teammate.

The project ran from May 19 to 28, 2025. The attributions follow the [commit history](https://github.com/uchicago-dsi-bayesianml/sp25-biscuit-anomaly-detection/commits/main).

## Getting started

There is no environment file. The Keras notebooks need TensorFlow, scikit-learn, matplotlib, Pillow and kagglehub; the PyTorch notebooks need torch, torchvision, transformers and scikit-learn, and were run in Google Colab on a GPU with data on Google Drive. Paths inside the notebooks point at those locations and need editing.

| Notebook | What it does |
|---|---|
| `Final Project VAE Model.ipynb` | Downloads the dataset with kagglehub and arranges it into train, validation and test folders with `ok` and `nok` subfolders; shows examples; an early VAE attempt that did not work (see below). |
| `Final Project Autoencoder.ipynb` | The convolutional autoencoder baseline at 256×256, with threshold selection and a classification report. |
| `EDA and VAE.ipynb` | Exploration, the rotation-group split, the Keras VAE at 64×64 trained on defect-free images, thresholding and evaluation. |
| `EDA, VAE and Clustering.ipynb` | The same, extended with K-means on the VAE latents to separate defect types, and a second VAE trained on defects only. |
| `DCGAN_biscuit_anomaly.ipynb` | The DCGAN at 64×64, latent-optimization anomaly scores, ROC analysis and the threshold from Youden's J. |
| `ViTVAE.ipynb` | The Vision Transformer VAE at 224×224, its training curve, thresholding, reconstructions and a t-SNE of the latent space. |

## Data

Data files are never committed to this repository.

- **Source:** the [Industry Biscuit (Cookie) dataset](https://www.kaggle.com/datasets/imonbilk/industry-biscuit-cookie-dataset) on Kaggle, by Šimon Bilík: 1,225 photographs of Tarallini biscuits from an industrial line, 474 defect-free and 751 defective across the three defect types, each provided at four rotations for 4,900 images of 256×256 pixels.
- **Terms:** GNU LGPL 3.0, as declared on Kaggle.
- **How to get it:** the first notebook downloads it with kagglehub, which needs a Kaggle account and API token.
- **Where to put it:** the notebooks expect `train/`, `val/` and `test/` folders with `ok` and `nok` subfolders; the first notebook builds them.

## Project structure

| Path | Contents |
|---|---|
| `REPORT.md` | The authors' write-up: dataset, preparation, each model with code and results, the final comparison, future work. |
| The six notebooks above | All code; there are no modules or scripts. |

## Method

**Framing.** Every model is trained on defect-free biscuits only and scored on reconstruction error. The threshold is chosen on a validation split, by F1 for the autoencoder and VAE and by Youden's J on the ROC curve for the GAN.

**Autoencoder.** Three convolution and pooling blocks down, three convolution and upsampling blocks back, mean squared error, 256×256 input.

**Variational autoencoder.** Three convolutions to a 64-unit dense layer, then a mean and a log-variance for the latent code, the reparameterization trick, and a mirrored transposed-convolution decoder; binary cross-entropy reconstruction plus the KL term; 64×64 input, trained for 50 epochs on defect-free images. K-means with three clusters on the latent means of the test biscuits the VAE flagged.

**DCGAN with latent optimization.** The generator and discriminator are the DCGAN implementation from Erik Linder-Norén's [PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN) repository, MIT licensed, with 34 of its 38 substantive lines reproduced verbatim; the authors' write-up presents it as their own architecture, and course staff added this credit. The GAN is trained on defect-free images for 100 epochs. The anomaly score follows the approach of AnoGAN (Schlegl et al., 2017): for each test image, a latent code is optimized for 200 Adam steps to make the generator reproduce the image, and the remaining mean squared error is the score. Scoring ran on a balanced set of 380 defect-free and 380 defective test images.

**Vision Transformer VAE.** A pretrained `google/vit-base-patch16-224` encoder, its patch embeddings split into a mean and log-variance map, and a four-layer transposed-convolution decoder; 100 epochs at 224×224. The class carries several unused parameters that suggest it was adapted from a template; course staff could not identify the source.

### What is Bayesian here

The VAE is the Bayesian piece. Its encoder is an amortized approximate posterior over a latent code, its decoder is the likelihood, and the loss is the negative evidence lower bound: a reconstruction term plus the KL divergence from a standard normal prior. Training it on defect-free biscuits only means the bound is tight for normal biscuits and loose for anything else, so reconstruction error is a crude stand-in for the model's evidence against a new image. The KL term is also what makes the latent space usable for the clustering: the codes of flagged biscuits are organized enough that incomplete shapes land together. The autoencoder is the same idea without the prior or the probabilistic encoder, which is why it is the right baseline. The GAN and the Vision Transformer VAE are generative models too, but the project uses them as reconstruction engines rather than as probability models: no likelihood or posterior is read off either.

What the project doesn't do is use the VAE's uncertainty. The reconstruction probability, averaging the decoder's likelihood over posterior samples of the code, is the Bayesian anomaly score for a VAE and would have used the model as a model.

## Results

As reported in the authors' write-up, on 400 test images for the first two models and the 760-image balanced set for the GAN.

| Model | Accuracy | Precision | Recall |
|---|---|---|---|
| Autoencoder (256×256) | 0.76 | 0.93 | 0.56 |
| VAE (64×64) | 0.79 | 0.72 | 0.95 |
| DCGAN with latent optimization (64×64) | 0.89 | 0.92 | 0.85 |
| Vision Transformer VAE (224×224) | not reported | 0.49 | 0.96 |

Precision and recall are for the defective class. The authors' error analysis shows the autoencoder catching the foreign objects and colored biscuits and missing the misshapen ones, and the VAE's clustering putting 44 of 48 incomplete-shape biscuits in one cluster. See the [caveats below](#what-to-learn-from-this-project) on where these numbers come from.

## What to learn from this project

Course staff added this section. The application is well chosen and the analysis around the models is the team's own; the points below are about attribution and about numbers.

- **Credit what you borrow.** The DCGAN's generator and discriminator are a public implementation reproduced nearly verbatim, and the latent-optimization score is AnoGAN's method, yet the write-up introduces the architecture as "our model" and cites neither. Both are fine to use. Using them without saying so is the one thing in this repository that a reader must not learn from. The credit now appears under [Method](#method).
- **Report the numbers your code produced.** The write-up's autoencoder table doesn't match the committed notebook, whose final report has a different threshold and reverses precision and recall. One sentence says 81 percent accuracy and the table under it says 0.76. The GAN's stated accuracy of 0.8933 doesn't follow from its own confusion matrix, which gives 0.888, and its stated recall of 0.8583 is 0.85 in the notebook. Small gaps, but a reader who tries to reproduce the table can't.
- **Compare on the same footing.** The autoencoder ran at 256×256, the VAE and GAN at 64×64, the Vision Transformer at 224×224. The GAN was scored on a balanced 760-image set and the others on the 400-image test set. The ranking may well hold, but the table doesn't establish it.
- **Say who did what.** No author is named anywhere in the write-up, and the fourth member exists only in a commit message.
- **Keep one copy of each notebook.** Two notebooks were byte-for-byte identical, a third is an earlier version of the fourth, an early VAE attempt that classifies everything as normal is still named "Final Project", and the notebooks carry 53 MB of outputs.

What the project does well: a leak-free split that respects the four rotations of each biscuit; a baseline, a Bayesian model and two alternatives on the same question; a threshold chosen on validation data; an error analysis that weighs the cost of each kind of mistake, where a screw matters more than a dent; a second question, which defects are tolerable, answered with the latent space; and a failed experiment written up with the right diagnosis instead of dropped.

## About this project

This project was built in Bayesian Machine Learning with Generative AI Applications (ADSP 32014), part of the Master of Science in Applied Data Science at the University of Chicago, in Spring 2025. It is shown here as an example for later cohorts.

## Changes by course staff

Course staff forked the authors' repository and made these edits. The authors' notebooks are unchanged.

- Renamed the authors' README to `REPORT.md`, unchanged, and added this README with the program banner, including the credits to PyTorch-GAN and AnoGAN that the write-up lacks.
- Removed `EDA, VAE, and Clustering.ipynb`, which was byte-for-byte identical to `EDA and VAE.ipynb`.
- Added a `.gitignore` for data and model files.

## License

The authors have not added a license, so their work is shown here under GitHub's terms for public repositories. Ask them before reusing it. The DCGAN code inside it is MIT licensed by its original author.
