"""tools/exl3_ref.py - the executable form of include/strata/kernels/exl3_expert.hpp: EXL3 trellis decode and the
integer-path routed expert every Strata device must reproduce bit for bit.

    python tools/exl3_ref.py --pack <exl3 pack> [--layers 0,4] [--experts 3] [--golden]   (self-checks)

  decode       the per-weight codebook index s of one matrix, from its trellis (integer and half-integer K)
  expert_ref   one token through one expert (gate/up -> SwiGLU -> down) in the spec's FP32 operations
  expert_f64   the same expert in float64 with exact weights (the error yardstick)
  --golden     decode against exllamav3's own reconstruct (needs its venv and a GPU): every weight must be exactly
               exllamav3's FP16 value
"""
import argparse
import json
import pathlib
import sys

import numpy as np

MUL1 = np.uint32(0x83DCD12D)
A = np.float32(0.00676727294921875)         # fp16 0x1eee
BIAS = np.float32(-10.3828125)              # fp16 0xc931
B = np.float32(1024.0) * A + BIAS           # -3.453125, exact
F = np.float32
ROLES = ("gate", "up", "down")


# ---- format -------------------------------------------------------------------------------------------------------

def _step_pos():
    p = np.arange(256)
    row = ((p // 8) % 4) * 2 + (p & 1) + 8 * ((p // 2) & 1)
    col = p // 32 + 8 * ((p // 4) & 1)
    return row, col


ROW, COL = _step_pos()


def step_ends(K2: int) -> np.ndarray:
    """End bit (exclusive) of each of a tile's 256 steps."""
    p = np.arange(256, dtype=np.int64)
    if K2 % 2 == 0:
        return (p + 1) * (K2 // 2)
    KA = K2 // 2
    return np.where(p % 2 == 0, (p // 2) * (2 * KA + 1) + KA, ((p + 1) // 2) * (2 * KA + 1))


def decode(trellis: np.ndarray, k: int, n: int, K2: int) -> np.ndarray:
    """s[k][n] (int32 0..1020): the codebook index of every weight of one matrix."""
    words_per_tile = 8 * K2 // 2 if K2 % 2 == 0 else 4 * K2   # 8 K uint32
    tiles = np.frombuffer(np.ascontiguousarray(trellis).tobytes(), np.uint32).reshape(k // 16, n // 16, words_per_tile)
    nbits = 32 * words_per_tile                               # 256 K
    # the bit stream of every tile, MSB-first, as a 0/1 array [tk][tn][nbits]
    w = tiles[..., :, None] >> np.arange(31, -1, -1, dtype=np.uint32)
    bits = (w & 1).astype(np.uint32).reshape(k // 16, n // 16, nbits)
    ends = step_ends(K2)
    idx = (ends[:, None] - 16 + np.arange(16)[None, :]) % nbits                 # [256][16] bit positions, MSB first
    state = np.zeros((k // 16, n // 16, 256), np.uint32)
    for b in range(16):
        state = (state << np.uint32(1)) | bits[..., idx[:, b]]
    x = state * MUL1                                                             # mod 2^32
    s = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + (x >> 24)
    out = np.zeros((k, n), np.int32)
    for tk in range(k // 16):
        for tn in range(n // 16):
            out[tk * 16 + ROW, tn * 16 + COL] = s[tk, tn]
    return out


def weights_fp16(s: np.ndarray) -> np.ndarray:
    """exllamav3's decoded W' (FP16): fp16(1024 + s) * A + BIAS in one FP16 FMA = the exact affine value, rounded."""
    return (s.astype(np.float64) * float(A) + float(B)).astype(np.float16)


class Pack:
    def __init__(self, d: pathlib.Path):
        self.d = d
        self.lines = {}
        for line in (d / "native_experts.txt").read_text().splitlines():
            if line and not line.startswith("#"):
                l, gt, dt, off, blob = (int(x) for x in line.split()[:5])
                self.lines[l] = (gt - 160, dt - 160, off, blob)
        self.mm = np.memmap(d / "experts.bin", np.uint8, "r")
        cfg = json.loads((d / "exl3.json").read_text())
        self.H, self.FF = 2560, 640
        del cfg

    def expert(self, l: int, e: int):
        """[(trellis, suh, svh, k, n, K2)] for gate, up, down."""
        gk2, dk2, off, blob = self.lines[l]
        at = off + e * blob
        out = []
        for r, (k, n, K2) in enumerate(((self.H, self.FF, gk2), (self.H, self.FF, gk2), (self.FF, self.H, dk2))):
            tb = (k // 16) * (n // 16) * 16 * K2
            tr = self.mm[at:at + tb]
            suh = np.frombuffer(self.mm[at + tb:at + tb + 2 * k].tobytes(), np.float16)
            svh = np.frombuffer(self.mm[at + tb + 2 * k:at + tb + 2 * k + 2 * n].tobytes(), np.float16)
            out.append((tr, suh, svh, k, n, K2))
            at += tb + 2 * k + 2 * n
        assert at == off + (e + 1) * blob
        return out


# ---- the spec's arithmetic ----------------------------------------------------------------------------------------

def had128(v: np.ndarray) -> np.ndarray:
    """Unnormalized 128-point Hadamard on each 128 values (FP32 butterflies, strides 1..64, (a + b, a - b))."""
    v = v.astype(F).reshape(-1, 128).copy()
    h = 1
    while h < 128:
        v = v.reshape(-1, 128 // (2 * h), 2, h)
        a, b = v[:, :, 0, :].copy(), v[:, :, 1, :].copy()
        v[:, :, 0, :] = a + b
        v[:, :, 1, :] = a - b
        v = v.reshape(-1, 128)
        h *= 2
    return v.reshape(-1)


def round_away(t: np.ndarray) -> np.ndarray:
    tr = np.trunc(t)
    fr = np.abs(t - tr)
    return np.where(fr >= F(0.5), tr + np.copysign(F(1.0), t), tr).astype(F)


def quant128(xh: np.ndarray):
    """int8 q, FP32 d per 128-block, Q per block."""
    v = xh.reshape(-1, 128)
    amax = np.abs(v).max(axis=1).astype(F)
    d = (amax / F(127.0)).astype(F)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(d[:, None] > 0, v / d[:, None], F(0.0)).astype(F)
    q = np.clip(round_away(t), -127, 127).astype(np.int32)
    return q, d, q.sum(axis=1)


def matvec_ref(x: np.ndarray, s: np.ndarray, suh: np.ndarray, svh: np.ndarray) -> np.ndarray:
    """One matrix, one token: steps 1-5 of the spec."""
    k, n = s.shape
    u = (x.astype(F) * suh.astype(F)).astype(F)
    q, d, Q = quant128(had128(u))
    I = np.einsum("ji,jic->jc", q.astype(np.int64), s.reshape(k // 128, 128, n).astype(np.int64))
    assert np.abs(I).max() < 2 ** 24
    acc = np.zeros(n, F)
    qs = F(0.0)
    for j in range(k // 128):
        acc = (acc + (d[j] * I[j].astype(F)).astype(F)).astype(F)
        qs = F(qs + F(d[j] * F(Q[j])))
    yp = ((A * acc).astype(F) + (B * qs)).astype(F)
    y = had128(yp)
    return ((y * F(1.0 / 128.0)).astype(F) * svh.astype(F)).astype(F)


def sexp(x: np.ndarray) -> np.ndarray:
    """include/strata/kernels/sexp.hpp in float32.  fma: the product is exact in long double (64-bit mantissa), the
    sum rounds once there and once to float32 - a double rounding that differs from a true fma only on ties far below
    the 2^-40 level (exl3_cpu_parity would flag one)."""
    x = x.astype(F)
    def fma(a, b, c):
        L = np.longdouble
        return (a.astype(L) * b.astype(L) + c.astype(L)).astype(F)
    k = np.rint((x * F(1.44269504)).astype(F)).astype(F)
    r = fma(k, np.full_like(x, F(-0.693145751953125)), x)
    r = fma(k, np.full_like(x, F(-1.42860677e-06)), r)
    p = np.full_like(x, F(1.98412698e-04))
    for c in (1.38888889e-03, 8.33333333e-03, 4.16666667e-02, 1.66666667e-01, 0.5, 1.0, 1.0):
        p = fma(p, r, np.full_like(x, F(c)))
    out = np.ldexp(p, k.astype(np.int32)).astype(F)
    out = np.where(x < F(-103.97208), F(0.0), out)
    return np.where(~(x < F(88.72284)), np.where(np.isnan(x), x, F(np.inf)), out).astype(F)


def expert_ref(x: np.ndarray, parts, s_cache=None) -> np.ndarray:
    """One token through gate/up -> SwiGLU (g / (1 + sexp(-g)) * up) -> down, the spec's FP32 operations."""
    ss = s_cache or [decode(tr, k, n, K2) for tr, _, _, k, n, K2 in parts]
    g = matvec_ref(x, ss[0], parts[0][1], parts[0][2])
    u = matvec_ref(x, ss[1], parts[1][1], parts[1][2])
    h = ((g / (F(1.0) + sexp(-g))).astype(F) * u).astype(F)
    return matvec_ref(h, ss[2], parts[2][1], parts[2][2])


def had_matrix() -> np.ndarray:
    Hm = np.array([[1.0]])
    while Hm.shape[0] < 128:
        Hm = np.block([[Hm, Hm], [Hm, -Hm]])
    return Hm


def matvec_f64(x, s, suh, svh):
    """y = H(H(x * suh) W') * svh / 128 in float64 with exact affine weights."""
    k, n = s.shape
    Hm = had_matrix()
    w = s.astype(np.float64) * float(A) + float(B)
    u = (x.astype(np.float64) * suh.astype(np.float64)).reshape(-1, 128) @ Hm.T
    y = u.reshape(-1) @ w
    return (y.reshape(-1, 128) @ Hm.T).reshape(-1) / 128.0 * svh.astype(np.float64)


def expert_f64(x, parts, s_cache=None):
    ss = s_cache or [decode(tr, k, n, K2) for tr, _, _, k, n, K2 in parts]
    g = matvec_f64(x, ss[0], parts[0][1], parts[0][2])
    u = matvec_f64(x, ss[1], parts[1][1], parts[1][2])
    h = g / (1.0 + np.exp(-g)) * u
    return matvec_f64(h, ss[2], parts[2][1], parts[2][2])


# ---- self-checks --------------------------------------------------------------------------------------------------

def golden(pack: Pack, layers, experts):
    """decode() against exllamav3's reconstruct (FP16 W' in the rotated basis)."""
    sys.path.insert(0, "/home/imdad/Work/inference/tools-3p/exl3/exllamav3")
    import torch
    from exllamav3.ext import exllamav3_ext as ext
    for l in layers:
        for e in range(experts):
            for r, (tr, suh, svh, k, n, K2) in zip(ROLES, pack.expert(l, e)):
                t = torch.from_numpy(np.frombuffer(tr.tobytes(), np.int16).reshape(k // 16, n // 16, 8 * K2).copy()).cuda()
                w = torch.empty((k, n), dtype=torch.half, device="cuda")
                ext.reconstruct(w, t, K2 / 2, False, True)
                want = w.cpu().numpy()
                got = weights_fp16(decode(tr, k, n, K2))
                bad = int((want.view(np.uint16) != got.view(np.uint16)).sum())
                print("golden layer %d expert %d %s (K %.1f): %d of %d weights differ" % (l, e, r, K2 / 2, bad, k * n),
                      flush=True)
                if bad:
                    raise SystemExit("decode does not match exllamav3")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True, type=pathlib.Path)
    ap.add_argument("--layers", default="0,4")
    ap.add_argument("--experts", type=int, default=2)
    ap.add_argument("--golden", action="store_true")
    ap.add_argument("--fixture", type=pathlib.Path, help="write x, h and y of --experts per layer, 2 inputs each, "
                    "for exl3_cpu_parity / exl3_replica")
    a = ap.parse_args()
    pack = Pack(a.pack)
    layers = [int(x) for x in a.layers.split(",")]
    if a.golden:
        golden(pack, layers, a.experts)
    rng = np.random.default_rng(0)
    if a.fixture:
        recs = []
        for l in layers:
            for e in range(a.experts):
                e2 = (e * 97 + l * 13) % 512          # spread the experts over the layer
                parts = pack.expert(l, e2)
                ss = [decode(tr, k, n, K2) for tr, _, _, k, n, K2 in parts]
                for scale in (0.5, 3.0):
                    x = (rng.standard_normal(pack.H) * scale).astype(F)
                    g = matvec_ref(x, ss[0], parts[0][1], parts[0][2])
                    u = matvec_ref(x, ss[1], parts[1][1], parts[1][2])
                    h = ((g / (F(1.0) + sexp(-g))).astype(F) * u).astype(F)
                    y = matvec_ref(h, ss[2], parts[2][1], parts[2][2])
                    recs.append(np.array([l, e2], np.int32).tobytes() + x.tobytes() + h.tobytes() + y.tobytes())
            print("fixture: layer %d done" % l, flush=True)
        a.fixture.write_bytes(np.array([len(recs), pack.H, pack.FF], np.int32).tobytes() + b"".join(recs))
        print("wrote %s: %d records" % (a.fixture, len(recs)))
        return 0
    for l in layers:
        for e in range(a.experts):
            parts = pack.expert(l, e)
            ss = [decode(tr, k, n, K2) for tr, _, _, k, n, K2 in parts]
            x = (rng.standard_normal(pack.H) * 0.5).astype(F)
            y = expert_ref(x, parts, ss)
            yt = expert_f64(x, parts, ss)
            rel = float(np.sqrt(((y - yt) ** 2).sum() / (yt ** 2).sum()))
            print("layer %d expert %d: spec vs float64 rel RMS %.5f (|y| rms %.4f)" % (l, e, rel, float(np.sqrt((yt ** 2).mean()))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
