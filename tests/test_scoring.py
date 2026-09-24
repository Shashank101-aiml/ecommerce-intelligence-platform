from ml.scoring.score_customers import risk_tier


def test_risk_tier_boundaries():
    assert risk_tier(0.95) == "High" and risk_tier(0.7) == "High"
    assert risk_tier(0.69) == "Medium" and risk_tier(0.4) == "Medium"
    assert risk_tier(0.39) == "Low" and risk_tier(0.0) == "Low"
