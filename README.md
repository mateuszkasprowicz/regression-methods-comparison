# Regression methods comparison | Three notebooks refactored into advanced Kedro workflow

## Overview

Initially, I developed three seperate notebooks to benchmark three ML techniques (linear regression, KNN, random forest) against three different datasets. After some time, I thought it might be a cool exercise to refactor them into a single Kedro workflow.

## Advanced Kedro functionalities used

- Modular pipelines
- Multiple namespace layers
- Hooks
    - external packages config
- Pipeline registry
    - direct pipeline registration
- Dataset factories
- Loading parameters in code