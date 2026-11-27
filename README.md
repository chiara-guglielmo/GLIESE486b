# Daneel — GLISE486b 

Good Luck Interpreting Every Strange Exoplanet

4(hours of sleep) 8(cups of coffee) 6(prayers) b(etter be worth it)

A teaching package for modelling exoplanet transits and building detection
pipelines. The repository contains the course example code, a runnable CLI,
and Sphinx documentation.

Documentation: https://chiara-guglielmo.github.io/GLIESE486b/

## Installation

### Prerequisites

- Python = 3.10.x

### Install from source

```bash
git clone https://github.com/chiara-guglielmo/GLIESE486b.git
cd GLIESE486b
pip install .
```

### Development installation

```bash
pip install -e '.[dev,docs]'
```

## Usage

After installation, you can run daneel from the command line:

```bash
daneel -i <input_file> --transit --output
```

### Command-line options

- `-i, --input`: Input YAML parameter file(s). If omitted, the default configuration for the selected operation is used.
- `-t, --transit`: Generate a transit light curve. If no input file is provided, the default GJ 486 b parameters are used.
- `-d, --detection {rf}`: Train and evaluate a Random Forest classifier for exoplanet detection.
- `-d, --detection {cnn}`: Train and evaluate a Convolutional Neural Network for exoplanet detection.
- `-p, --predict CSV`: Classify light curves from a CSV file using the pretrained Convolutional Neural Network, without retraining the model.
- `-g, --gan`: Train a Generative Adversarial Network (GAN) and generate synthetic planetary light curves.
- `-o, --output`: Enable additional output saving for supported operations in the current working directory.

### Examples

```bash
# Generate a transit light curve with default parameters
daneel -t

# Generate a transit light curve with custom parameters and save the plot
daneel -i examples/params.yaml -t -o

# Train a Random Forest classifier with default parameters
daneel -d rf

# Train a Random Forest classifier with custom parameters
daneel -i examples/RF_parameters.yaml -d rf

# Train a Convolutional Neural Network with default parameters
daneel -d cnn

# Train a Convolutional Neural Network with custom parameters
daneel -i examples/CNN_parameters.yaml -d cnn

# Classify light curves using the pretrained CNN
daneel -p tess_data.csv

# Classify light curves and save the predictions
daneel -p tess_data.csv -o

# Train the GAN using the default configuration
daneel -g

# Train the GAN using the default configuration and save all results
daneel -g -o

# Train the GAN using a custom configuration
daneel -i examples/GAN_parameters.yaml -g

# Train the GAN using a custom configuration and save all results
daneel -i examples/GAN_parameters.yaml -g -o

# Classify GAN-generated light curves using the pretrained CNN
daneel -p GAN_generated_curves.csv -o
```

## Input File Format

The input format depends on the selected operation.

### YAML configuration files

Transit generation, classifier training and GAN training use YAML
configuration files containing the parameters required for the analysis.

Example configuration files are available in the `examples/` directory.

### Default configurations

When the `-i` or `--input` option is omitted, the following default
configurations are used:

| Operation | Default configuration |
|-----------|-----------------------|
| Transit generation | Default GJ 486 b parameters |
| Random Forest | `examples/RF_parameters.yaml` |
| CNN | `examples/CNN_parameters.yaml` |
| GAN | `examples/GAN_parameters.yaml` |

The default configuration files are located relative to the project
root, independently of the current working directory.

When training a classifier or a GAN, the dataset path must be specified
in the configuration file.

Missing configuration files or invalid dataset paths raise an error.

## Development

Run the test suite and style checks with:

```bash
pytest
ruff check .
```

Build the documentation with `make -C docs html`.

## License

This project is licensed under the GNU General Public License v3.0.

## Author

Tiziano Zingales (tiziano.zingales@unipd.it), 

Angelica Plazzotta (angelica.plazzotta@studenti.unipd.it), 

Chiara Guglielmo (chiara.guglielmo.1@studenti.unipd.it), 

Mattia Saccon (mattia.saccon@studenti.unipd.it)
