"""
Star Counter — detection pipeline prototype
Pipeline: load -> grayscale -> background subtraction -> sigma-threshold
          point-source detection (DAOStarFinder) -> annotate -> score.

This will feel familiar: it's PIV particle detection pointed at the sky.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.background import Background2D, MedianBackground
from PIL import Image


def detect_stars(image_path, fwhm=3.0, sigma_threshold=5.0):
    """Detect point sources in a night-sky image.

    Returns (sources_table, grayscale_array).
    """
    img = Image.open(image_path).convert("L")
    data = np.array(img, dtype=float)

    # Background subtraction — removes light-pollution gradients,
    # exactly like background subtraction in flow images.
    bkg = Background2D(data, box_size=64, filter_size=3,
                       bkg_estimator=MedianBackground())
    data_sub = data - bkg.background

    # Robust noise estimate via sigma clipping
    _, median, std = sigma_clipped_stats(data_sub, sigma=3.0)

    # Point-source detection: matched filter for Gaussian PSFs
    finder = DAOStarFinder(fwhm=fwhm, threshold=sigma_threshold * std)
    sources = finder(data_sub - median)
    return sources, data


def sky_score(n_stars, image_area_px):
    """Toy sky-quality score: star density mapped to a 0-100 scale.
    v2: calibrate against Bortle scale using exposure metadata (EXIF)."""
    density = n_stars / image_area_px * 1e6  # stars per megapixel
    return float(min(100, density * 2.5))


def annotate(image_path, out_path):
    sources, data = detect_stars(image_path)
    n = 0 if sources is None else len(sources)

    fig, ax = plt.subplots(figsize=(8, 8 * data.shape[0] / data.shape[1]))
    ax.imshow(data, cmap="gray", origin="upper",
              vmin=np.percentile(data, 5), vmax=np.percentile(data, 99.5))
    if sources is not None:
        ax.scatter(sources["xcentroid"], sources["ycentroid"],
                   s=80, facecolors="none", edgecolors="#5DCAA5",
                   linewidths=1.0)
    score = sky_score(n, data.size)
    ax.set_title(f"{n} stars detected — sky score {score:.0f}/100",
                 fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return n, score


def make_synthetic_sky(path, n_stars=350, size=(900, 1200), seed=7):
    """Generate a synthetic night sky for testing: Gaussian PSF stars on a
    noisy background with a light-pollution gradient."""
    rng = np.random.default_rng(seed)
    h, w = size
    yy, xx = np.mgrid[0:h, 0:w]

    # Light-pollution gradient (bright toward one corner) + read noise
    sky = 18 + 25 * (xx / w) * (yy / h) + rng.normal(0, 3, (h, w))

    xs = rng.uniform(5, w - 5, n_stars)
    ys = rng.uniform(5, h - 5, n_stars)
    # Star brightness roughly follows a power law (few bright, many faint)
    amps = 25 + 230 * rng.power(0.35, n_stars)
    fwhm = 3.0
    s = fwhm / 2.355

    for x0, y0, a in zip(xs, ys, amps):
        x0i, y0i = int(x0), int(y0)
        r = 8
        ys_l, ys_h = max(0, y0i - r), min(h, y0i + r)
        xs_l, xs_h = max(0, x0i - r), min(w, x0i + r)
        py, px = np.mgrid[ys_l:ys_h, xs_l:xs_h]
        sky[ys_l:ys_h, xs_l:xs_h] += a * np.exp(
            -((px - x0) ** 2 + (py - y0) ** 2) / (2 * s ** 2))

    sky = np.clip(sky, 0, 255).astype(np.uint8)
    Image.fromarray(sky).save(path)
    return n_stars


if __name__ == "__main__":
    truth = make_synthetic_sky("/home/claude/synthetic_sky.png")
    n, score = annotate("/home/claude/synthetic_sky.png",
                        "/home/claude/detected_stars.png")
    print(f"Ground truth: {truth} stars injected")
    print(f"Detected:     {n} stars ({n/truth*100:.1f}% recovery)")
    print(f"Sky score:    {score:.0f}/100")
