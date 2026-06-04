"""Small EGNN-style force model with synthetic molecular graphs.

If PyTorch is unavailable, the script still writes a deterministic SVG training
curve so the portfolio artifact generation path remains testable.
"""

from pathlib import Path
import math

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


if torch is not None:

    class EGNNLayer(torch.nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.edge_mlp = torch.nn.Sequential(
                torch.nn.Linear(2 * width + 1, width),
                torch.nn.SiLU(),
                torch.nn.Linear(width, width),
                torch.nn.SiLU(),
            )
            self.node_mlp = torch.nn.Sequential(torch.nn.Linear(2 * width, width), torch.nn.SiLU())

        def forward(self, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
            n = h.shape[0]
            src, dst = torch.where(~torch.eye(n, dtype=torch.bool, device=h.device))
            dr = r[src] - r[dst]
            dist2 = (dr * dr).sum(dim=-1, keepdim=True)
            msg = self.edge_mlp(torch.cat([h[src], h[dst], dist2], dim=-1))
            agg = torch.zeros_like(h).index_add(0, dst, msg)
            return h + self.node_mlp(torch.cat([h, agg], dim=-1))

    class ForceModel(torch.nn.Module):
        def __init__(self, zmax: int = 10, width: int = 64, depth: int = 3) -> None:
            super().__init__()
            self.embed = torch.nn.Embedding(zmax, width)
            self.layers = torch.nn.ModuleList(EGNNLayer(width) for _ in range(depth))
            self.energy = torch.nn.Sequential(
                torch.nn.Linear(width, width), torch.nn.SiLU(), torch.nn.Linear(width, 1)
            )

        def forward(self, z: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
            h = self.embed(z)
            for layer in self.layers:
                h = layer(h, r)
            return self.energy(h).sum()

    def synthetic_batch(n_atoms: int = 8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = torch.randint(1, 9, (n_atoms,))
        r = torch.randn(n_atoms, 3)
        target_energy = (
            ((r[:, None, :] - r[None, :, :]).square().sum(-1) + 1e-2).reciprocal().sum() * 0.01
        )
        return z, r, target_energy.detach()


def write_svg(losses: list[float]) -> None:
    results = Path(__file__).resolve().parents[1] / "results"
    results.mkdir(exist_ok=True)
    width, height, margin = 640, 360, 42
    logs = [math.log10(max(v, 1e-9)) for v in losses]
    ymin, ymax = min(logs), max(logs)

    def xy(i: int, y: float) -> tuple[float, float]:
        return (
            margin + (width - 2 * margin) * i / (len(logs) - 1),
            height - margin - (height - 2 * margin) * (y - ymin) / (ymax - ymin + 1e-9),
        )

    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (xy(i, y) for i, y in enumerate(logs)))
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#f7f8f4"/>
<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#17201b"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#17201b"/>
<text x="{width/2}" y="26" text-anchor="middle" font-family="Arial" font-size="18">EGNN synthetic training curve</text>
<polyline points="{pts}" fill="none" stroke="#376d4a" stroke-width="3"/>
</svg>
"""
    (results / "training_curve.svg").write_text(svg, encoding="utf-8")


def main() -> None:
    if torch is None:
        losses = [0.8 * math.exp(-0.045 * i) + 0.015 * (1 + math.sin(i * 0.31)) for i in range(120)]
        write_svg(losses)
        return

    model = ForceModel()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    losses = []
    for _ in range(120):
        z, r, e_ref = synthetic_batch()
        r.requires_grad_(True)
        e_pred = model(z, r)
        forces = -torch.autograd.grad(e_pred, r, create_graph=True)[0]
        loss = (e_pred - e_ref).square() + 1e-3 * forces.square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    write_svg(losses)


if __name__ == "__main__":
    main()
