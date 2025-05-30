# XiP: XSPEC Interactive Plotting Tool
**- By Aditya Pandya**
## Overview
XiP is an interactive plotting tool designed for XSPEC-based spectral analysis. It is used to visualize the effects of different XSPEC models and their parameters by creating interactive plots in Jupyter Notebooks with the help of `ipywidgets`. 

> **Note**: This tool is primarily designed with eROSITA data in mind, but other X-ray data may require some modifications.

## Features
- **Interactive Plotting**: Visualize spectral data, different XSPEC models and residuals interactively using sliders for model parameters.
- **Model Configuration**: XiP can be run in three modes: `src`, `bkg`, and `src_bkg`, allowing users to configure the source and background models separately or together.
- **Parameter Management**: Ability to freeze parameters, tie parameters and fit the model using XSPEC's. In `src_bkg` and `bkg` modes, users can add filter wheel closed model as vary its normalization

## Requirements

Install all dependencies using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```
> **Note**: Ensure XSPEC is installed and properly configured on your system.

## Usage

Follow the instructions in the [XiP_demo.ipynb](XiP_demo.ipynb) notebook to configure data and models and plot the spectrum. The notebook provides a step-by-step guide on how to set up the data directory, source and background files, and model parameters.

You can also find a detailed explanation of the internal parameters and their usage on the [XiP GitHub Wiki](https://github.com/adipandya/XiP/wiki).

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any bugs or feature requests.

## Author's Note
This tool was built because I found it difficult to visualize the effects of different XSPEC models and their parameters. 

I hope you find it useful for your spectral analysis needs. If you have any suggestions for improvements or new features, please feel free to reach out at my email id: aditya.pandya@astro.uni-tuebingen.de.

#### Disclaimer 
This tool is a work in progress and may have bugs or limitations. Please report any issues you encounter, and feel free to suggest improvements or new features.
