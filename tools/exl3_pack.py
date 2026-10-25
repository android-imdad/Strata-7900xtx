"""tools/exl3_pack.py - a native pack whose routed experts are EXL3 (exllamav3) trellis weights.

    python tools/exl3_pack.py --exl3 <exllamav3 quant dir> --base <native pack> --out <new pack>

Only the routed experts change: the base pack (tools/iq_pack.py) supplies everything else - dense.bin, index.txt,
conversions.json and the tokenizer are linked from it, and the engine still runs its GGUF (--native) for the dense
weights and the router.  The experts come from the EXL3 quant (turboderp's Qwen3.8-Flash-Next-exl3, mul1 codebook)
and go to experts.bin, one blob per (layer, expert), three roles back to back:

    [gate: trellis | suh | svh][up: trellis | suh | svh][down: trellis | suh | svh]

    trellis  exllamav3's int16 [k/16][n/16][16 K] tiles exactly as stored (K bits a weight: 2.5 = 40 int16 a tile)
    suh, svh FP16 [k], [n]: the input and output channel scales (y = H(H(x * suh) W') * svh, H the 128-point
             Hadamard over each 128 values - include/strata/kernels/exl3_expert.hpp)

Every part is a multiple of 256 bytes at these widths (k, n in {2560, 640}), so the roles stay aligned.  The layer's
format is a type id per role in native_experts.txt v6: EXL3 at K bits = 160 + 2 K (K 2.5 -> 165, 3 -> 166).
"""
import argparse
import json
import os
import pathlib
import struct
import sys

import numpy as np

MUL1 = 0x83DCD12D          # the mul1 codebook's multiplier, stored per tensor as an int32
ROLES = ("gate", "up", "down")
EXL3_TYPE0 = 160           # type id = 160 + 2 K


def exl3_type(K2: int) -> int:
    return EXL3_TYPE0 + K2


class Shards:
    """The quant's safetensors, tensor name -> (file, absolute offset, dtype, shape)."""

    def __init__(self, d: pathlib.Path):
        self.where = {}
        for f in sorted(d.glob("*.safetensors")):
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                head = json.loads(fh.read(n))
            for k, v in head.items():
                if k != "__metadata__":
                    a, b = v["data_offsets"]
                    self.where[k] = (f, 8 + n + a, b - a, v["dtype"], v["shape"])
        self.maps = {}

    def raw(self, name: str) -> np.ndarray:
        f, off, size, _, _ = self.where[name]
        m = self.maps.get(f)
        if m is None:
            m = self.maps[f] = np.memmap(f, np.uint8, "r")
        return m[off:off + size]

    def shape(self, name: str):
        return self.where[name][4]

    def dtype(self, name: str):
        return self.where[name][3]


def role_geometry(sh: Shards, p: str):
    """(k, n, K2) of one EXL3 matrix, checked against its trellis shape and codebook."""
    t = sh.shape(p + ".trellis")
    if sh.dtype(p + ".trellis") != "I16" or len(t) != 3:
        raise SystemExit("%s.trellis: not an int16 [k/16][n/16][16K] tensor" % p)
    k, n, words = 16 * t[0], 16 * t[1], t[2]
    if words % 8:
        raise SystemExit("%s: %d int16 a tile is not a whole number of half-bits" % (p, words))
    K2 = words // 8                   # int16 a tile = 16 K  ->  2 K = words / 8
    if not (2 <= K2 <= 16) or (K2 % 2 and K2 > 7):
        raise SystemExit("%s: bitrate %.1f is not one EXL3 takes (1..8, or 1.5/2.5/3.5)" % (p, K2 / 2))
    mul1 = int(np.frombuffer(bytes(sh.raw(p + ".mul1")), np.int32)[0]) & 0xFFFFFFFF
    if mul1 != MUL1:
        raise SystemExit("%s: codebook multiplier 0x%08X is not mul1's 0x%08X" % (p, mul1, MUL1))
    if sh.shape(p + ".suh") != [k] or sh.shape(p + ".svh") != [n]:
        raise SystemExit("%s: suh/svh are not [%d], [%d]" % (p, k, n))
    if k % 128 or n % 128:
        raise SystemExit("%s: %d x %d is not whole 128-value Hadamard blocks" % (p, k, n))
    return k, n, K2


def role_bytes(k: int, n: int, K2: int) -> int:
    return (k // 16) * (n // 16) * 16 * K2 + 2 * k + 2 * n   # trellis (16 K2 bytes a tile) + suh + svh


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exl3", required=True, type=pathlib.Path, help="exllamav3 quant directory")
    ap.add_argument("--base", required=True, type=pathlib.Path, help="native pack to take the dense weights from")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--layers", default="", help="comma list: pack only these layers' rows (tests); all by default")
    ap.add_argument("--check", action="store_true", help="re-read every blob part and compare it with the source")
    a = ap.parse_args()

    cfg = json.loads((a.exl3 / "config.json").read_text())
    q = cfg.get("quantization_config", {})
    if q.get("quant_method") != "exl3" or q.get("codebook") != "mul1":
        raise SystemExit("%s is not an EXL3 mul1 quant" % a.exl3)
    tc = cfg.get("text_config", cfg)
    n_layers, n_expert = int(tc["num_hidden_layers"]), int(tc["num_experts"])
    H, FF = int(tc["hidden_size"]), int(tc["moe_intermediate_size"])
    sh = Shards(a.exl3)
    pre = "model.language_model.layers.%d.mlp.experts.%d.%s_proj"

    # geometry per layer (every expert of a layer must share it)
    geo = []
    for l in range(n_layers):
        g = [role_geometry(sh, pre % (l, 0, r)) for r in ROLES]
        if g[0][:2] != (H, FF) or g[1][:2] != (H, FF) or g[2][:2] != (FF, H) or g[0][2] != g[1][2]:
            raise SystemExit("layer %d: expert geometry %s is not gate/up %dx%d, down %dx%d at one K" % (l, g, H, FF, FF, H))
        geo.append(g)
    want = set(range(n_layers)) if not a.layers else {int(x) for x in a.layers.split(",")}

    a.out.mkdir(parents=True, exist_ok=True)
    for name in ("dense.bin", "index.txt", "conversions.json", "tokenizer"):
        src, dst = (a.base / name).resolve(), a.out / name
        if src.exists() and not dst.exists():
            os.symlink(src, dst)

    lines, offset = [], 0
    for l, g in enumerate(geo):
        blob = sum(role_bytes(*x) for x in g)
        if blob % 256:
            raise SystemExit("layer %d: blob of %d bytes is not 256-aligned" % (l, blob))
        lines.append("%d %d %d %d %d" % (l, exl3_type(g[0][2]), exl3_type(g[2][2]), offset, blob))
        offset += blob * n_expert
    total = offset

    path = a.out / "experts.bin"
    with open(path, "r+b" if path.exists() else "w+b") as out:
        out.truncate(total)
        offset = 0
        for l, g in enumerate(geo):
            blob = sum(role_bytes(*x) for x in g)
            if l in want:
                for e in range(n_expert):
                    parts = []
                    for r in ROLES:
                        p = pre % (l, e, r)
                        if role_geometry(sh, p) != g[ROLES.index(r)]:
                            raise SystemExit("%s: geometry differs from expert 0's" % p)
                        parts += [sh.raw(p + ".trellis"), sh.raw(p + ".suh"), sh.raw(p + ".svh")]
                    data = b"".join(bytes(x) for x in parts)
                    assert len(data) == blob
                    out.seek(offset + e * blob)
                    out.write(data)
                print("layer %2d: K %.1f / %.1f, %d B a blob" % (l, g[0][2] / 2, g[2][2] / 2, blob), flush=True)
            offset += blob * n_expert

    if a.check:
        with open(path, "rb") as f:
            mm = np.memmap(f, np.uint8, "r")
            offset = 0
            for l, g in enumerate(geo):
                blob = sum(role_bytes(*x) for x in g)
                if l in want:
                    for e in range(n_expert):
                        at = offset + e * blob
                        for r in ROLES:
                            p = pre % (l, e, r)
                            for s in (".trellis", ".suh", ".svh"):
                                src = sh.raw(p + s)
                                if not np.array_equal(mm[at:at + len(src)], src):
                                    raise SystemExit("check: %s%s differs in experts.bin" % (p, s))
                                at += len(src)
                offset += blob * n_expert
        print("check: every part matches its source tensor", flush=True)

    head = ("# strata native experts v6: layer gu_type d_type offset blob_bytes (n_expert %d, total %d; EXL3 experts in "
            "experts.bin from %s, type 160 + 2K: blob = [trellis|suh|svh] x gate, up, down)\n" % (n_expert, total, a.exl3.name))
    (a.out / "native_experts.txt").write_text(head + "".join(x + "\n" for x in lines))
    (a.out / "exl3.json").write_text(json.dumps({
        "source": str(a.exl3.resolve()), "quantization_config": q, "base_pack": str(a.base.resolve()),
        "layers": [{"layer": l, "K": [x[2] / 2 for x in g]} for l, g in enumerate(geo)]}, indent=1) + "\n")
    print("wrote %s (%.2f GB experts.bin)" % (a.out, total / 1e9))
    return 0


if __name__ == "__main__":
    sys.exit(main())
