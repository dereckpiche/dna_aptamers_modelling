# Predicting DNA Free Energy with Neural Networks

Can neural networks learn the relationship between a DNA sequence and its simulated folding free energy?

This project compares a multilayer perceptron (MLP), an LSTM and a Transformer on one million randomly generated DNA sequences of 30 bases. The target values come from NUPACK thermodynamic calculations, rather than experimental measurements.

## Research workflow

1. **Generate the dataset.** Pair random DNA sequences with NUPACK free-energy labels.
2. **Encode the sequences.** Use one-hot representations for the MLP and LSTM, and learned nucleotide embeddings with positional encoding for the Transformer.
3. **Train the models.** Use a random 90%/10% training–evaluation split and mean squared error (MSE) as the regression loss.
4. **Compare learning curves.** Examine prediction error as training progresses and identify overfitting.
5. **Analyse prediction errors.** Compare the larger Transformer's predictions across free-energy quartiles to identify where performance deteriorates.

### Free-energy distribution

![Distribution of NUPACK free-energy labels for one million DNA sequences](doc/images/MFE_nupack.png)

The labels are concentrated toward less negative free energies. Sequences in the low-energy tail are less common, making performance across the full distribution worth examining.

## Model comparison

The [final report](doc/rapport_final.pdf) records the following approximate evaluation MSE values:

| Model | Reported MSE |
| --- | ---: |
| MLP | 0.94 |
| LSTM | 0.91 |
| Transformer | 0.15 |
| Larger Transformer | 0.12 |

![Evaluation loss across MLP, LSTM and Transformer training](doc/images/comparaison.png)

The Transformer achieved substantially lower error in these experiments. The models differ in size and training duration, so this comparison does not isolate architecture alone.

### Transformer learning curve

![Transformer training and evaluation mean squared error](doc/images/transformer.png)

Training error continues to fall while evaluation error levels off, illustrating the importance of monitoring generalisation.

The larger Transformer's best reported MSE was approximately 0.12. Its later saved-checkpoint analysis produced an MSE of 0.1564, not 0.12. In that analysis, MSE ranged from 0.3500 in the lowest-energy quartile to 0.0347 in the highest-energy quartile.

## Code and analysis

- [MLP baseline](src/mlpbaseline.py)
- [LSTM notebook](src/LSTM.ipynb)
- [Transformer notebook](src/transformer.ipynb) and [larger Transformer](src/large_transformer.ipynb)
- [Prediction-error analysis](src/results/AptAnalysis.ipynb)
- [Final report and methodology](doc/rapport_final.pdf)

These are historical experimental results. The evaluation split was monitored during training, so it is not an untouched final test. Learning NUPACK-generated labels does not establish accuracy on experimental free energies or predict binding affinity.

---

# Welcome to our aptamers Gibbs free energy prediction project!

## The "doc" folder
The doc folder contains all the .tex source files of different reports we had to routinely furnish. The "images" subfolder contains all the graphics included in said reports.

## The "src" folder
The src folder contains all the Jupyter and Python scripts we used in order to create our datasets and train our models. Most of our experiments were written in Jupyter and are self-descriptive. You only need to press the "run all cells" button and it should work. When the training of our models is finished, the scripts generate serialized files containing the weights of our models. These saved models were used in some other files in order to provide additionnal results for the analysis section of our report. Inside the  "results" subfolder, you will find some code we used in order to analyse and generate more results.
Inside the "data" subfolder, you will find our dataset in *Json* format and the code we wrote in order to generate it with the NUPACK library. To run the dataset generation code on your own, you will need to install the NUPACK library on your machine https://www.nupack.org/download/overview. 
