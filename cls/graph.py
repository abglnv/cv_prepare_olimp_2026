"""
Plot helper for the architecture playground.

    from graph import graph
    graph(total_metrics + [exp_metrics])

`graph()` takes a list of metrics dicts (each one returned by `train(model, name)`)
and draws two panels side by side:

  left   val top-1 (%) vs epoch, one curve per experiment  -> see the learning curves
  right  best val top-1 (%) vs experiment name (a line)     -> see the ranking at a glance

Each metrics dict is expected to carry "name" (set by train), "epoch", "val_top1",
"final_val_top1", "best_val_top1", and "params_M".

    from graph import visualize_batch
    visualize_batch(process_batch(x))     # x is [-1, 1] normalised [B, C, H, W]
"""
import matplotlib.pyplot as plt


def visualize_batch(images, titles=None, figsize=(5, 5)):
    """Show the first 4 images of a batch in a 2x2 grid.

    `images` is a float tensor [B, C, H, W] normalised to [-1, 1] (the range the
    augmentation hook works in). Values are mapped back to [0, 1] just for display.
    """
    x = images[:4].detach().to("cpu").float()
    x = ((x + 1.0) * 0.5).clamp(0, 1)               # [-1, 1] -> [0, 1]
    x = x.permute(0, 2, 3, 1).numpy()               # [B, C, H, W] -> [B, H, W, C]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    for i, ax in enumerate(axes.ravel()):
        if i < x.shape[0]:
            img = x[i]
            ax.imshow(img[..., 0], cmap="gray") if img.shape[-1] == 1 else ax.imshow(img)
            if titles is not None and i < len(titles):
                ax.set_title(str(titles[i]), fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def graph(metrics_list, figsize=(13, 4.5)):
    """Overlay val-top1 curves and plot best val-top1 per experiment."""
    if isinstance(metrics_list, dict):       # tolerate a single metrics dict
        metrics_list = [metrics_list]
    if not metrics_list:
        print("graph: nothing to plot (empty list)")
        return

    names = [m.get("name") or f"exp{i}" for i, m in enumerate(metrics_list)]

    fig, ax = plt.subplots(1, 2, figsize=figsize)

    # --- left: per-epoch val top-1 curves, one per experiment ---------------
    for name, m in zip(names, metrics_list):
        label = f"{name} (best {m['best_val_top1']:.1f}%, {m['params_M']:.1f}M)"
        ax[0].plot(m["epoch"], m["val_top1"], marker="o", ms=3, label=label)
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("val top-1 (%)")
    ax[0].set_title("val top-1 over training")
    ax[0].legend(fontsize=8)
    ax[0].grid(True, alpha=0.3)

    # --- right: best val top-1 per experiment (line plot) -------------------
    bests = [m["best_val_top1"] for m in metrics_list]
    ax[1].plot(names, bests, marker="o", color="tab:red")
    ax[1].set_xlabel("experiment")
    ax[1].set_ylabel("best val top-1 (%)")
    ax[1].set_title("best val top-1 by experiment")
    ax[1].grid(True, alpha=0.3)
    for x, y in zip(names, bests):
        ax[1].annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                       xytext=(0, 6), ha="center", fontsize=8)
    plt.setp(ax[1].get_xticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    plt.show()
