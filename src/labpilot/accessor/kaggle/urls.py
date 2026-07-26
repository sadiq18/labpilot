"""Canonical Kaggle web URLs for competitions and kernels."""


def competition_submissions_url(slug: str) -> str:
    return f"https://www.kaggle.com/competitions/{slug}/submissions"


def kernel_notebook_url(owner: str, kernel_slug: str, version: int | None = None) -> str:
    base = f"https://www.kaggle.com/code/{owner}/{kernel_slug}"
    if version is not None:
        return f"{base}/versions/{version}"
    return base


def parse_kernel_ref(kernel_ref: str) -> tuple[str, str]:
    """Split `owner/slug` into components."""
    ref = kernel_ref.strip().strip("/")
    if ref.startswith("code/"):
        ref = ref.removeprefix("code/")
    parts = ref.split("/")
    if len(parts) != 2:
        raise ValueError(f"Kernel ref must be owner/slug, got {kernel_ref!r}")
    return parts[0], parts[1]
