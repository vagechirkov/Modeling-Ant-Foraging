# Modeling-Ant-Foraging-Seminar


## Installation

###  Check that `Python` and `pip` are installed and on the path
```bash
python3 --version
```

### Create a virtual environment (built-in `venv` module)

```bash
# Pick any folder name; “venv” or “.venv” are common
python3 -m venv .venv
```

### Activate the environment

| Platform                 | Command                      |
| ------------------------ | ---------------------------- |
| **macOS / Linux**        | `source .venv/bin/activate`  |
| **Windows (PowerShell)** | `.venv\Scripts\Activate.ps1` |
| **Windows (cmd.exe)**    | `.venv\Scripts\activate.bat` |


### Install packages

```bash
pip --version  #  check that `pip` is installed and on the correct path
pip install -U "mesa[all]"
pip install notebook ipykernel
```

### Register this venv as a named Jupiter kernel

```bash
python3 -m ipykernel install --user --name foraging_ants_seminar \
       --display-name "Modeling Ant Foraging Seminar"
```

### Launch the Notebook server
```bash
jupyter notebook          # or:  jupyter lab
```

A browser tab opens at http://localhost:8888.

Stop the server with **Ctrl-C** in the terminal.
