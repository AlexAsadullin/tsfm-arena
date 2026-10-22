from __future__ import annotations

import math

import pytest
import torch

from tsfm_arena.training import losses


def _mask(x: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(x, dtype=torch.bool)


def test_pairwise_logistic_closed_form() -> None:
    s = torch.tensor([[0.5, -0.5]])
    y = torch.tensor([[1.0, 0.0]])
    assert losses.pairwise_logistic_loss(s, y, _mask(s), 1.0).item() == pytest.approx(math.log1p(math.exp(-1)))
    zero = torch.zeros(1, 4)
    loss = losses.pairwise_logistic_loss(zero, torch.arange(4.0)[None], _mask(zero), 1.0)
    assert loss.item() == pytest.approx(math.log(2))


def test_listnet_matches_manual() -> None:
    s = torch.tensor([[1.0, 0.0, -1.0]])
    y = torch.tensor([[0.3, 0.1, -0.4]])
    z = (y - y.mean()) / y.std(unbiased=False)
    expected = -(torch.softmax(z, 1) * torch.log_softmax(s, 1)).sum()
    assert losses.listnet_loss(s, y, _mask(s), 1.0).item() == pytest.approx(expected.item(), rel=1e-5)


def test_soft_spearman_tends_to_minus_one_when_aligned() -> None:
    y = torch.tensor([[0.1, 0.5, -0.2, 0.3, 0.0]])
    assert losses.soft_spearman_loss(y * 100, y, _mask(y), 0.01).item() == pytest.approx(-1.0, abs=1e-3)
    assert losses.soft_spearman_loss(-y * 100, y, _mask(y), 0.01).item() == pytest.approx(1.0, abs=1e-3)
    ranks = losses.soft_ranks(torch.tensor([[3.0, 1.0, 2.0]]), torch.ones(1, 3, dtype=torch.bool), 1e-3)
    assert ranks.tolist()[0] == pytest.approx([2.5, 0.5, 1.5])


@pytest.mark.parametrize("name", ["pairwise", "listnet", "spearman"])
def test_rank_losses_ignore_masked_assets_and_train(name: str) -> None:
    fn = losses.rank_loss(name)  # type: ignore[arg-type]
    torch.manual_seed(0)
    y = torch.randn(4, 6)
    mask = torch.ones(4, 6, dtype=torch.bool)
    mask[:, -1] = False
    s = torch.randn(4, 6, requires_grad=True)
    base = fn(s, y, mask, 1.0)
    perturbed = s.detach().clone()
    perturbed[:, -1] += 100.0
    assert fn(perturbed, y, mask, 1.0).item() == pytest.approx(base.item(), abs=1e-5)
    opt = torch.optim.SGD([s], lr=1.0)
    for _ in range(50):
        opt.zero_grad()
        loss = fn(s, y, mask, 1.0)
        loss.backward()
        opt.step()
    assert fn(s, y, mask, 1.0).item() < base.item()


def test_pinball_and_mse_masks() -> None:
    pred = torch.tensor([[0.0, 1.0], [5.0, 5.0]])
    target = torch.tensor([1.0, 100.0])
    mask = torch.tensor([True, False])
    assert losses.pinball_loss(pred, target, [0.1, 0.9], mask).item() == pytest.approx((0.1 * 1 + 0.0) / 2)
    assert losses.mse_loss(torch.tensor([0.0, 9.0]), torch.tensor([2.0, 0.0]), mask).item() == 4.0
    with pytest.raises(KeyError):
        losses.rank_loss("none")
