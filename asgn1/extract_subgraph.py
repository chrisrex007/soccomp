from pathlib import Path


# Keep only the edges whose endpoints are both odd, i.e. drop every node
# whose id is divisible by 2.
def extract_subgraph():
    raw_path = Path("subgraphs") / "cit-HepTh.txt"
    out_path = Path("subgraphs") / "cite-phy.elist"

    Path("subgraphs").mkdir(exist_ok=True)

    with open(raw_path, "r") as fin, open(out_path, "w") as fout:
        for line in fin:
            if line.startswith("#"):
                continue

            parts = line.strip().split()
            if len(parts) < 2:
                continue

            src = int(parts[0])
            dst = int(parts[1])

            if src % 2 != 0 and dst % 2 != 0:
                fout.write(f"{src}\t{dst}\n")

    print(f"Subgraph written to {out_path}")


if __name__ == "__main__":
    extract_subgraph()
