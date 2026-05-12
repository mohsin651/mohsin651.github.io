---
title: "Welcome — Sample Post with LaTeX Support"
description: "A sample blog post showing how to write articles with LaTeX math formulas."
date: 2026-05-01
tags: ["meta", "latex"]
---

Welcome to my blog! This is a sample post demonstrating that **LaTeX math** works perfectly here.

## Inline Math

Einstein's famous equation: $E = mc^2$

The cross-entropy loss for binary classification: $\mathcal{L} = -[y \log(\hat{y}) + (1-y)\log(1-\hat{y})]$

## Block Math

The softmax function used in classification:

$$
\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}} \quad \text{for } i = 1, \dots, K
$$

Something more complex is here:

$$
\begin{aligned}
\mathcal{L} &= \frac{1}{2} g^{\mu\nu} \nabla_\mu \phi \nabla_\nu \phi 
- \frac{1}{2} m^2 \phi^2 
- \frac{\lambda}{4!} \phi^4 \\
&\quad + \bar{\psi}(i \gamma^\mu D_\mu - m)\psi 
- \frac{1}{4} F_{\mu\nu}F^{\mu\nu} \\
&\quad + \int \frac{d^4k}{(2\pi)^4} \,
\frac{e^{-ikx}}{k^2 - m^2 + i\epsilon}
+ \sum_{n=1}^{\infty} \frac{(-1)^n}{n!} 
\left( \int d^4x \, \mathcal{L}_{\text{int}} \right)^n
\end{aligned}
$$

The attention mechanism from Transformers:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

## Code Blocks

```python
import torch
import torch.nn.functional as F

def softmax(x):
    return F.softmax(x, dim=-1)
```

## Lists, links, and more

- Standard markdown works
- [Links](https://astro.build) work
- **Bold** and *italic* work
- All the usual formatting

That's it! Just create new `.md` files in `src/content/blog/` to add new posts.
