"""Extract the cite-phy subgraph from the raw SNAP cit-HepTh dataset.

Reads the downloaded edge list, removes all nodes whose IDs are divisible
by 2, and writes the remaining edges to subgraphs/cite-phy.elist.
"""

from pathlib import Path


def extract_subgraph() -> None:
    """Filter edges to keep only those where both endpoints are odd."""

    raw_path = Path("subgraphs") / "cit-HepTh.txt"
    out_path = Path("subgraphs") / "cite-phy.elist"

    # Ensure the output directory exists.
    Path("subgraphs").mkdir(exist_ok=True)

    with open(raw_path, "r") as fin, open(out_path, "w") as fout:
        for line in fin:
            # Skip comment lines that start with '#'.
            if line.startswith("#"):
                continue

            parts = line.strip().split()
            if len(parts) < 2:
                continue

            src = int(parts[0])
            dst = int(parts[1])

            # Keep edges where both endpoints have odd IDs (not divisible by 2).
            if src % 2 != 0 and dst % 2 != 0:
                fout.write(f"{src}\t{dst}\n")

    print(f"Subgraph written to {out_path}")


if __name__ == "__main__":
    extract_subgraph()
