# Vision Transformer (ViT) - TensorFlow Implementation

This is a TensorFlow implementation of the paper:  
**"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"**  
by Dosovitskiy et al., published by Google Research in 2020.

Link to the paper: https://arxiv.org/abs/2010.11929

##  Paper Summary

ViT applies the Transformer architecture, originally designed for NLP, to image classification by:
- Splitting images into fixed-size patches (e.g., 16x16)
- Flattening and embedding them as input tokens
- Processing the tokens with a standard Transformer encoder
- Using the output of a `[CLS]` token for classification

##  Key Contributions
- Introduced the first pure Transformer model for vision.
- Achieved SOTA on ImageNet when pretrained on large datasets (e.g., JFT-300M).
- Demonstrated the importance of scale for Transformer success in vision.

---

##  Features

- ViT from scratch: patch embedding, positional encoding, transformer encoder
- Supports ViT-Ti, ViT-S, ViT-B, ViT-L (configurable)...
- Trains on CIFAR-10, CIFAR-100, or ImageNet subsets
- Implements training loop, validation, and metrics tracking

---

##  Installation
