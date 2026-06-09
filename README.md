# Enhancing MMEA with UA and PSM

This repository contains experimental code for enhancing multimodal multi-objective evolutionary algorithms (MMEAs) with UA and PSM strategies. The project combines Python-based enhancement scripts with MATLAB/PlatEMO-based algorithm runners and benchmark problem code.

## Project Structure

```text
.
├── Enhanced_CMMO.py
├── Enhanced_CoMMEA.py
├── Enhanced_HREA.py
├── Enhanced_MMEAPSL.py
├── Enhanced_MMEAWI.py
├── cluster_multimodal.py
├── model.py
├── problem.py
├── data/
│   ├── PF/
│   └── PS/
└── MMEAs/
    ├── CMMO/
    ├── CoMMEA/
    ├── HREA/
    ├── MMEAPSL/
    └── MMEAWI/
```

## Main Components

### Python Enhancement Scripts

The `Enhanced_*.py` files are the main Python scripts for running the enhanced versions of different MMEAs:

- `Enhanced_CMMO.py`
- `Enhanced_CoMMEA.py`
- `Enhanced_HREA.py`
- `Enhanced_MMEAPSL.py`
- `Enhanced_MMEAWI.py`

The shared Python modules provide reusable functionality:

- `problem.py`: problem-related definitions and utilities.
- `model.py`: model components used by the enhancement framework.
- `cluster_multimodal.py`: clustering utilities for multimodal solution analysis.

### MATLAB / PlatEMO-Based Code

The `MMEAs/` directory contains MATLAB code modified from PlatEMO. These files include benchmark problem modifications and scripts for calling/running the corresponding algorithms.

Each subdirectory corresponds to one baseline MMEA:

- `MMEAs/CMMO/`
- `MMEAs/CoMMEA/`
- `MMEAs/HREA/`
- `MMEAs/MMEAPSL/`
- `MMEAs/MMEAWI/`

The MATLAB scripts follow the naming pattern:

```text
run_<algorithm>_<problem>.m
```

For example:

```text
MMEAs/CMMO/run_cmmo_mmf1.m
MMEAs/HREA/run_hrea_omnitest.m
MMEAs/MMEAWI/run_mmeawi_sympart.m
```

These scripts are intended to be used inside a MATLAB environment with PlatEMO available.

### Data

The `data/` directory stores reference data for benchmark evaluation:

- `data/PF/`: Pareto front reference data.
- `data/PS/`: Pareto set reference data.

Files are named by benchmark problem, for example:

```text
data/PF/mmf1_pf.dat
data/PS/mmf1_ps.dat
```

## Requirements

### Python

Use a Python scientific computing environment. The project depends on common scientific computing packages and also requires `pymoo` and MATLAB Engine for Python, because the Python scripts call MATLAB code during the experiment workflow.

- NumPy
- SciPy
- scikit-learn
- Matplotlib
- pymoo
- PyTorch, if the model implementation uses neural components
- MATLAB Engine for Python

### MATLAB

The MATLAB code requires:

- MATLAB
- PlatEMO
- MATLAB Engine for Python

Before running scripts in `MMEAs/`, make sure PlatEMO is added to the MATLAB path. When running the Python enhancement scripts, MATLAB Engine for Python is used to call MATLAB code from Python.

## Usage

### Running Enhanced Python Experiments

Run one of the enhanced algorithm scripts from the project root:

```bash
python Enhanced_CMMO.py
python Enhanced_CoMMEA.py
python Enhanced_HREA.py
python Enhanced_MMEAPSL.py
python Enhanced_MMEAWI.py
```

Adjust experiment settings directly inside the corresponding script if needed.

### Running MATLAB / PlatEMO Experiments

Open MATLAB, add PlatEMO and this repository to the MATLAB path, then run the desired script. For example:

```matlab
run('MMEAs/CMMO/run_cmmo_mmf1.m')
```

or open the script in MATLAB and run it interactively.

## Benchmarks

The repository includes scripts and reference data for several multimodal multi-objective benchmark problems, including:

- MMF1
- MMF1_e
- MMF1_z
- MMF2
- MMF3
- MMF4
- MMF5
- MMF6
- MMF7
- MMF8
- MMF9
- MMF14
- MMF14_a
- Omni-test
- SYM-PART

## Notes

- The MATLAB code under `MMEAs/` is based on modified PlatEMO code and is mainly used for problem definitions and algorithm invocation.
- The Python scripts implement the enhanced experimental workflow for the selected MMEAs.
- Reference PF and PS files are stored separately to support evaluation against known Pareto fronts and Pareto sets.

