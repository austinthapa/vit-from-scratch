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

### Feeding Image into Transformer
```
┌───────────────────────────────┐
│       Input Image (H×W×C)     │
│       224 × 224 × 3           │
└─────────────┬─────────────────┘
              │ Split into non-overlapping patches
              │ each of size 16×16 patches
┌─────────────▼──────────────────┐
│ Number of patches N = H*W / P² │
│ N = 196 patches (14×14)        │
└─────────────┬──────────────────┘
              │ Flatten each patch:
              │ Each patch → 16×16×3 = 768 dims
┌─────────────▼────────────────┐
│ Patch vectors: shape(N × 768)│
│ (196 × 768)                  │
└─────────────┬────────────────┘
              │ Linear projection (trainable)
              │ Project 768 → D (e.g. 768)
┌─────────────▼─────────────┐
│ Patch embeddings (N × D)  │
│ (196 × 768)               │
└─────────────┬─────────────┘
              │ Prepend [CLS] token (1 × D)
              │ Single learnable vector
┌─────────────▼─────────────────────┐
│ Input sequence: (N+1 × D)         │
│ [CLS],Patch_1,Patch_2,...,Patch_N │
│ (197 × 768)                       │
└─────────────┬─────────────────────┘
              │ Add positional embeddings (N+1 × D)
              │ Learned vectors indicating position
┌─────────────▼────────────────┐
│ Final input to Transformer   │
│ Sequence shape: (197 × 768)  │
└─────────────┬────────────────┘
              │ Transformer encoder layers process
              │ the sequence via self-attention
┌─────────────▼───────────────┐
│ Transformer output sequence │
│ (197 × 768)                 │
│                             │
│ Use output at [CLS] token   │
│ as image representation     │
└─────────────┬───────────────┘
              │ Pass CLS token output to MLP head
              │ for classification logits
┌─────────────▼─────────────┐
│  Final prediction logits  │
└───────────────────────────┘

```

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

