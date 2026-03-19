#!/bin/bash

optimum-cli export onnx \
  --model BAAI/bge-reranker-v2-m3 \
  --task text-classification \
  --dtype fp16 \
  --optimize O3 \
  ./bge-reranker-v2-m3
