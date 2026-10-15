# Cloud experiment notebooks

`phase4_colab_kaggle.ipynb` runs the first real AdaptiveDepth static-route
experiment on a Colab or Kaggle GPU.

Before running it:

1. Push the desired repository commit to GitHub.
2. If the repository is private, create a fine-grained GitHub token with
   read-only access to this repository's contents, add it to Colab or Kaggle
   Secrets as `GITHUB_TOKEN`, and grant the notebook access to it. Never paste
   the token into a code cell.
3. Enable a GPU runtime.
4. On Kaggle, enable Internet access so GitHub and Hugging Face are reachable.
5. Run the cells from top to bottom.
6. Download `phase4_artifacts.zip` before the runtime is discarded.

The Phase 4 command is resumable. Keep the `experiments/phase4` directory when
reconnecting to the same GPU/software environment. It intentionally rejects a
resume on different hardware because those latency measurements are not
comparable.
