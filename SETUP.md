## Setup and install

This testing suite uses Poetry for dependency management.

If you don't already have poetry installed, please follow their guide [here](https://python-poetry.org/docs/).

### Activate the virtual enviornment

```bash
poetry env info #Look for executable path

source .venv/bin/activate # Linux
# OR
source .venv/bin/python3.12 # Windows
```

### Install deps
```bash
poetry install --no-root #No root is used since .venv lives in project directory.
```

### Adding new packages.
Poetry will resolve all package dependecy conflicts for us. However, be careful as this sometimes changes the python version requirements. Be considerate when altering. 

```bash
poetry add {some_pypi_package}
```

### Configure VS code to use interpreter
```bash
ctrl + shift + p
Python: Select Interpreter # Select this
```
And enter the interpreter path found in .venv (same as source command) 

## Running

Once the virtual enviornment is activated and all dependencies installed you're free to run the script using: 
```bash
python3 MechInterpFeatureMeaning.py
```