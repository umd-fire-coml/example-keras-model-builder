# Keras Model Builder Example

![example workflow](https://github.com/umd-fire-coml/example-keras-model-builder/actions/workflows/run_all_tests.yaml/badge.svg)

This package contains the following files:

```bash
├── src
│   ├── datagen.py
├── test
│   ├── test_datagen.py
├── .github
│   ├── workflows
│       ├── run_all_tests.yaml
├── requirements.txt
├── test-requirements.txt
└── .gitignore
```

## Source Directory

The src directory contains all the source code files for your project. In this example, there is one file - model.py, which creates a simple model. You will edit this file to create a simple model for one of the sklearn or tensorflow datasets, e.g. iris, mnist, etc.

## Test Directory

The test directory contains all the unit tests for our source code files (test_model.py).

## .github Directory

The .github/workflows directory contains the different workflows that we want to run using GitHub Actions. The run_all_tests.yaml file runs all of our tests as separate jobs any time we push, create a pull request, or manually run on GitHub.

## Requirements

The requirements.txt file and test-requirements.txt file contain the required packages for the code and tests respectively.

## Testing Locally

To run all tests locally, make sure pytest is installed. Then, run
```bash
python -m pytest
```
in the root directory.

To run a single test, make sure pytest is installed. Them, run
```bash
python -m pytest test/test_file_you_want_to_run.py
```

## Testing On GitHub Actions

To view the tests that have run and their logs,

1. Go to your GitHub Repository
2. Click on Actions
3. View the most recent run of your workflow or manually run your workflow