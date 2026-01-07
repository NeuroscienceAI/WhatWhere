Code and data used to reproduce the study: What and where manifolds emerge and align with perception in deep neural network models of sound localization.

We visualize audio embeddings and the low-dimensional structure of sound representations in models trained for sound localization.

We analyze two open-source datasets: one on gerbil sound localization (https://openreview.net/forum?id=t7xYNN7RJC#discussion
) (Ralph E. Peterson, Aramis Tanelus, ..., Dan H. Sanes, Alex H. Williams, 2024, NeurIPS) (Figures 2–3; Supplementary Figures 1–4), and one on human sound localization (https://www.nature.com/articles/s41467-024-54700-5
) (Mark R. Saddler and Josh H. McDermott, 2024, Nature Communications) (Figures 5–8; Supplementary Figures 5–10; Supplementary Table 1). The datasets can be downloaded from https://vclbenchmark.flatironinstitute.org/
 and https://drive.google.com/drive/folders/1YgC7x6Ot84XZInlSyHK-9NQ0jhhGUS2z
.

We train two 1D CNNs (5- and 10-layer) for the gerbil sound localization task from scratch. We download the pretrained models (PyTorch) from https://github.com/msaddler/phaselocknet_torch
 and https://drive.google.com/drive/folders/1qcW_Z5iX45dObOqbiD_Yo1dLqvVyiqoH
.

We appreciate the original authors for releasing their datasets and pretrained models with detailed annotations.
