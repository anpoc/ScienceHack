# invoice_batch_dataset.py
from __future__ import annotations

import random
import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple

from pypdf import PdfReader, PdfWriter
from torch.utils.data import Dataset


class InvoiceBatchDataset(Dataset):
    """
    Each __getitem__(idx) returns
        (tmp_pdf_path: Path, label_list: List[int])

    * tmp_pdf_path – path to a (possibly cached) PDF stored in
                     <system-tmp>/invoice_batch_dataset/<train|test>/0001.pdf …
    * label_list   – first page of every original PDF gets label 1, the rest 0.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        min_n: int = 5,
        max_n: int = 20,
        size: int = 100,
    ):
        root = Path(root)
        assert split in {"train", "test"}, "split must be 'train' or 'test'"
        assert 1 <= min_n <= max_n

        self.min_n = min_n
        self.max_n = max_n
        self.size = size
        self.split = split

        # ------------------------------------------------------------------
        # Collect every directory that represents ONE original PDF
        # ------------------------------------------------------------------
        if split == "test":
            folders = [root / "challenge"]
        else:  # train
            folders = [p for p in root.iterdir() if p.is_dir() and p.name != "challenge"]

        pdf_dirs: list[Path] = []
        for top in folders:
            pdf_dirs.extend([p for p in top.iterdir() if p.is_dir()])

        if not pdf_dirs:
            raise RuntimeError(f"No PDF sub-folders found beneath: {folders}")

        self.pdf_dirs = pdf_dirs

        # ------------------------------------------------------------------
        # Prepare tmp/<train|test> folder inside the system temp directory
        # ------------------------------------------------------------------
        self._tmp_root = Path(tempfile.gettempdir()) / "invoice_batch_dataset"
        self.tmp_dir = self._tmp_root / split
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------
    # PyTorch Dataset API
    # ----------------------------------------------------------------------
    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[Path, List[int]]:
        """
        The idx is now used to generate deterministic filenames (0001.pdf, …)
        and to seed a private RNG so the same idx always yields identical data.
        """
        if idx < 0 or idx >= self.size:
            raise IndexError
        rng = random.Random(idx)                 # deterministic per-sample RNG
        n = rng.randint(self.min_n, self.max_n)  # how many PDFs to combine
        chosen_dirs = rng.sample(self.pdf_dirs, n)

        labels: List[int] = []
        page_files: List[Path] = []

        for pdf_dir in chosen_dirs:
            pages = sorted(pdf_dir.glob("*.pdf"))
            if not pages:
                continue

            labels.append(1)                     # first page of a new PDF
            labels.extend([0] * (len(pages) - 1))
            page_files.extend(pages)

        # ------------------------------------------------------------------
        # Determine output path   tmp/<split>/0001.pdf, 0002.pdf, …
        # ------------------------------------------------------------------
        out_path = self.tmp_dir / f"{idx + 1:04d}.pdf"

        # ------------------------------------------------------------------
        # Create PDF only if it doesn't already exist
        # ------------------------------------------------------------------
        if not out_path.exists():
            writer = PdfWriter()
            for pf in page_files:
                reader = PdfReader(pf, strict=False)
                writer.add_page(reader.pages[0])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "wb") as fh:
                writer.write(fh)

        return out_path, labels

    # ----------------------------------------------------------------------
    # House-keeping: clean up tmp files when the Dataset is gone
    # ----------------------------------------------------------------------
    def __del__(self) -> None:
        try:
            shutil.rmtree(self._tmp_root, ignore_errors=True)
        except Exception:
            # Avoid throwing during interpreter shutdown
            pass