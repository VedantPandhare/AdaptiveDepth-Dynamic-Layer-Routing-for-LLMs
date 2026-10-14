# Cloud experiment notebooks

`phase4_colab_kaggle.ipynb` runs the first real AdaptiveDepth static-route
experiment on a Colab or Kaggle GPU.

Before running it:

1. Push the desired repository commit to GitHub.
2. Enable a GPU runtime.
3. On Kaggle, enable Internet access so Hugging Face can download the model.
4. Run the cells from top to bottom.
5. Download `phase4_artifacts.zip` before the runtime is discarded.

The Phase 4 command is resumable. Keep the `experiments/phase4` directory when
reconnecting to the same GPU/software environment. It intentionally rejects a
resume on different hardware because those latency measurements are not
comparable.
