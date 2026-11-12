"""
Tests for the analytic ROI formula that replaces beam-search horizon guessing.

Value of capturing a planet at ETA with prod p and remaining R steps:
    V = w_prod * p * (R - eta)      # production we gain post-capture
      + w_deny * p * (R - eta)      # enemy production we deny (enemy targets only)
      - w_send * ships_sent         # opportunity cost of ships dispatched
      + w_cap  * ships_captured     # ships we gain from the garrison (friendly takeover)

Key properties we MUST satisfy:
1. Near beats far for same prod (compounding)
2. High prod beats low prod at same distance
3. Value is zero (or negative) if ETA >= remaining (fleet can never land)
4. Enemy planets worth ~2x neutral of same prod (gain + deny)
"""
import pytest
import sys
sys.path.insert(0, 'submission')


def analytic_value(prod, eta, remaining, ships_sent, ships_captured,
                   is_enemy=False,
                   w_prod=1.0, w_deny=1.0, w_send=1.0, w_cap=0.5):
    """Closed-form value of a planet capture."""
    steps_held = max(0.0, remaining - eta)
    gain       = w_prod * prod * steps_held
    deny       = (w_deny * prod * steps_held) if is_enemy else 0.0
    cost       = w_send * ships_sent
    cap        = w_cap  * ships_captured
    return gain + deny + cap - cost


class TestAnalyticROI:

    def test_near_beats_far_same_prod(self):
        """Near planet (eta=5) must score higher than far (eta=40) with equal prod."""
        r = 200
        v_near = analytic_value(prod=2, eta=5,  remaining=r, ships_sent=8, ships_captured=5)
        v_far  = analytic_value(prod=2, eta=40, remaining=r, ships_sent=8, ships_captured=5)
        assert v_near > v_far, f"near={v_near:.1f} far={v_far:.1f}"

    def test_high_prod_beats_low_prod_same_distance(self):
        """Higher production planet must score higher at same ETA."""
        r = 200
        v_lo = analytic_value(prod=1, eta=10, remaining=r, ships_sent=6, ships_captured=3)
        v_hi = analytic_value(prod=4, eta=10, remaining=r, ships_sent=8, ships_captured=3)
        assert v_hi > v_lo, f"hi={v_hi:.1f} lo={v_lo:.1f}"

    def test_zero_value_when_eta_exceeds_remaining(self):
        """Fleet that can't land before game ends has non-positive value."""
        v = analytic_value(prod=5, eta=201, remaining=200, ships_sent=1, ships_captured=0)
        assert v <= 0, f"expected <=0, got {v:.1f}"

    def test_enemy_worth_double_neutral_same_prod(self):
        """Enemy planet: we gain prod AND deny it — roughly 2x a neutral."""
        r = 100
        v_neutral = analytic_value(prod=2, eta=10, remaining=r,
                                   ships_sent=12, ships_captured=5, is_enemy=False)
        v_enemy   = analytic_value(prod=2, eta=10, remaining=r,
                                   ships_sent=12, ships_captured=5, is_enemy=True)
        assert v_enemy > v_neutral * 1.5, (
            f"enemy={v_enemy:.1f} should be >1.5× neutral={v_neutral:.1f}"
        )

    def test_far_high_prod_beats_near_very_low_prod(self):
        """Far planet (eta=15) with prod=10 beats near (eta=2) with prod=1 —
        only when the production ratio dominates the timing penalty."""
        r = 50
        v_near = analytic_value(prod=1, eta=2,  remaining=r, ships_sent=4,  ships_captured=2)
        v_far  = analytic_value(prod=10, eta=15, remaining=r, ships_sent=12, ships_captured=4)
        assert v_far > v_near, f"far={v_far:.1f} near={v_near:.1f}"

    def test_send_cost_penalises_large_fleets(self):
        """All else equal, sending more ships lowers value."""
        r = 100
        v_cheap  = analytic_value(prod=2, eta=10, remaining=r, ships_sent=6,  ships_captured=5)
        v_costly = analytic_value(prod=2, eta=10, remaining=r, ships_sent=30, ships_captured=5)
        assert v_cheap > v_costly, f"cheap={v_cheap:.1f} costly={v_costly:.1f}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
