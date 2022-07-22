
# Neural Preference Isolation Forest

This project aims at extending Preference Isolation Forest algorithm with the use of Neural Networks.
It has been developed as my Master Thesis and this is the code to run the framework.



## Installation

The code can be installed on LINUX as a python module as follows

```bash
  ./install.sh
```

## Project structure

At the root, the folders are structured in this way

    .
    ├── src                   # Source code files
    ├── main                  # Folder from which run the tests
    ├── LICENSE               # License file
    ├── README.md             # This readme
    ├── install.sh            # Installation file
    └── requirements.txt      # List of packages to install

Digging into `src` folder, it is composed as follows

    .
    ├── files                                   # All project source code
    │   ├── classes                             # Models to use with PIF
    │   │   ├── base_models.py
    │   │   ├── __init__.py
    │   │   ├── mss_model.py
    │   │   ├── neural_models.py
    │   │   └── self_organizing_maps.py
    │   ├── functions                           # Generic function to extract results from tests
    │   │   ├── __init__.py
    │   │   ├── results_extractors.py
    │   │   └── results_rocs_maker.py
    │   ├── __init__.py
    │   ├── main                                # Folder from which start the project
    │   │   ├── anomaly_detection_tests.py
    │   │   ├── parameters.json
    │   ├── pif                                 # Folder containing PIF source code and Voronoi Tesselation
    │   │   ├── __init__.py
    │   │   ├── pif.py
    │   │   ├── voronoi_iforest.py
    │   │   ├── voronoi_inode.py
    │   │   └── voronoi_itree.py
    │   └── utils                               # Folder containing generic utility functions
    │       ├── constants.py
    │       ├── dataset_creator.py
    │       ├── __init__.py
    │       └── utility_functions.py
    ├── __init__.py
    ├── notebooks                               # Folder containing all prototyping Jupyter notebooks
    │       └── ...
    ├── setup.py                                # File for installing the module
    └── test                                    # Test folder
        └── ...

## Run Locally

Clone the project

```bash
  git clone https://github.com/catonzio/Neural-PreferenceIsolation.git
```

Install python module and dependencies

```bash
  ./install.sh
```

Go to main folder

```bash
  cd main/
```

After setting parameters in ```parameters.json```, run main file
```bash
    python main.py
```


## Authors

- [@catonzio](https://www.github.com/catonzio)

