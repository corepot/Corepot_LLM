#!/bin/bash
echo "Exporting model to ONNX format..."
python app/export_onnx.py --model-path models/best_model.pth --output models/model.onnx



