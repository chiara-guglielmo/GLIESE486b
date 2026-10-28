# Daneel — Computational Astrophysics 2026–27

A teaching package for modelling exoplanet transits and building detection
pipelines. The repository contains the course example code, a runnable CLI,
and Sphinx documentation.


## Installation

### Prerequisites

- Python = 3.10

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
daneel -i <input_file> --transit
```

### Command-line options

- `-i, --input`: Input parameter file (or it use G486b values)
- `-t, --transit`: Generate a transit light curve from the YAML input

### Examples

```bash
# Run exoplanet detection
daneel -i examples/params.yaml -t

```

## Input File Format

The input file should be a YAML file containing the necessary parameters for the analysis.

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