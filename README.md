# minterpy-levelsets

A Python library for performing numerical differential geometry on smooth closed surfaces based on Global Polynomial Level Sets (GPLS). [^1]

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Development team](#develpment-team)
- [Contributing](#contributing)
- [License](#license)


## Background

Starting with a pointset representation of a surface, GPLS can be used to approximate a broad class of smooth surfaces as affine algebraic varieties. With this polynomial representation, differential-geometric quantities like mean and Gauss curvature can be efficiently and accurately computed. This compressed representation significantly reduces the computational cost of 3d surface simulations.


## Install

Since this implementation is a prototype, we currently only provide the installation by self-building from source. We recommend to using `git` to get the `minterpy-levelsets` source:

```bash
git clone https://codebase.helmholtz.cloud/interpol/minterpy-levelsets.git
```

Switch to the `conda` or `venv` virtual environment of your choice where you would like to install the library.

From within the environment, install using [pip],

```bash
pip install [-e] .
```

where the flag `-e` means the package is directly linked
into the python site-packages of your Python version.

You **must not** use the command `python setup.py install` to install `minterpy`,
as you cannot always assume the files `setup.py` will always be present
in the further development of `minterpy`.


## Usage

Documentation is a WIP. Please refer to the example Jupyter notebooks in the `examples/` directory to get started with the library.


## Development team

### Main code development
- Sachin Krishnan Thekke Veettil (MPI CBG/TU Dresden) <sthekke@mpi-cbg.de>
- Gentian Zavalani (HZDR/CASUS) <g.zavalani@hzdr.de>

### Mathematical foundation
- Michael Hecht (HZDR/CASUS) <m.hecht@hzdr.de>

### Acknowledgement
- Uwe Hernandez Acosta (HZDR/CASUS)
- Damar Wicaksono (HZDR/CASUS)
- Minterpy development team

## Contributing

[Open an issue](https://codebase.helmholtz.cloud/interpol/minterpy-levelsets/-/issues) or submit PRs.


## License

[MIT](LICENSE)

[^1]: [Veettil, Sachin K. Thekke, Gentian Zavalani, Uwe Hernandez Acosta, Ivo F. Sbalzarini, and Michael Hecht. "Global Polynomial Level Sets for Numerical Differential Geometry of Smooth Closed Surfaces." arXiv preprint arXiv:2212.11536 (2022)] (https://arxiv.org/abs/2212.11536).
