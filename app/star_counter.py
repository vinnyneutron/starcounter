"""
Star Counter — detection pipeline
Pipeline: load -> grayscale -> background subtraction -> sigma-threshold
          point-source detection (DAOStarFinder) -> annotate -> score.
"""

import io

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.background import Background2D, MedianBackground
from PIL import Image

MAX_DIM = 1600


def _load_grayscale(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img.load()
    img = img.convert("L")

    if max(img.size) > MAX_DIM:
        scale = MAX_DIM / max(img.size)
        img = img.resize((max(1, round(img.width * scale)),
                          max(1, round(img.height * scale))))

    return np.array(img, dtype=float)


def detect_stars(data, fwhm=3.0, sigma_threshold=5.0):
    """Detect point sources in a grayscale image array.

    Returns the photutils sources table (or None if none found).
    """
    box_size = 64 if min(data.shape) >= 64 else max(8, min(data.shape) // 2)
    bkg = Background2D(data, box_size=box_size, filter_size=3,
                        bkg_estimator=MedianBackground())
    data_sub = data - bkg.background

    # Robust noise estimate via sigma clipping
    _, median, std = sigma_clipped_stats(data_sub, sigma=3.0)

    # Point-source detection: matched filter for Gaussian PSFs
    finder = DAOStarFinder(fwhm=fwhm, threshold=sigma_threshold * std)
    sources = finder(data_sub - median)
    return sources


def sky_score(n_stars, image_area_px):
    """Toy sky-quality score: star density mapped to a 0-100 scale."""
    density = n_stars / image_area_px * 1e6  # stars per megapixel
    return float(min(100, density * 2.5))


def analyze_image(image_bytes: bytes):
    """Run the full detection pipeline on raw uploaded image bytes.

    Returns (n_stars, score, annotated_png_bytes).
    """
    data = _load_grayscale(image_bytes)
    sources = detect_stars(data)
    n = 0 if sources is None else len(sources)
    score = sky_score(n, data.size)

    fig, ax = plt.subplots(figsize=(8, 8 * data.shape[0] / data.shape[1]))
    ax.imshow(data, cmap="gray", origin="upper",
              vmin=np.percentile(data, 5), vmax=np.percentile(data, 99.5))
    if sources is not None:
        ax.scatter(sources["xcentroid"], sources["ycentroid"],
                   s=80, facecolors="none", edgecolors="#5DCAA5",
                   linewidths=1.0)
    ax.set_title(f"{n} stars detected — sky score {score:.0f}/100", fontsize=13)
    ax.axis("off")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return n, score, buf.getvalue()
