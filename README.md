# VRP Visual Attractiveness Prediction

A machine learning project that predicts which parts of vehicle routing problem (VRP) solutions will be modified by human planners, with explainable AI visualizations.

## Overview

This project trains neural networks to:
- Predict if/where a VRP route will be modified by planners
- Generate heatmaps showing model attention on route segments
- Evaluate explainability through various visualization techniques

## Key Components

- **Training Pipeline**: `train.py` - Train models with configurable architectures
- **Visualization**: `visualize.py` - Generate heatmaps and analysis
- **Optimization Pipeline**: `optimized_vrp_pipeline.py` - End-to-end route processing
- **Java Solver**: `MSH/` - Gurobi-based VRP optimization engine

## Quick Start

### Training
```bash
python train.py model.name=vgg batch_size=32 model_params.epochs=50
```

### Visualization  
```bash
python visualize.py model.load=true heatmap.method=gradcam
```

## Project Structure

- `src/models/` - Neural network architectures (VGG, ResNet, DeiT, etc.)
- `src/visualization/` - Heatmap generation and overlay utilities  
- `src/pipeline/` - Data processing and optimization workflows
- `config/` - Hydra configuration files
- `checkpoints/` - Trained model weights
- `MSH/` - Java-based VRP solver implementation
