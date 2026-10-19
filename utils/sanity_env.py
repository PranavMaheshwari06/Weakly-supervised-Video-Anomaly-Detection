from __future__ import annotations

import cv2
import numpy as np
import torch
from rich import print


def main() -> None:
    print("[bold]Environment sanity[/bold]")
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("opencv:", cv2.__version__)
    print("numpy:", np.__version__)


if __name__ == "__main__":
    main()
