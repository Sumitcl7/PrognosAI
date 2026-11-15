
import unittest

from src.decision.maintenance_engine_v2 import (
    evaluate_maintenance_v2,
)


class TestMaintenanceEngineV2(unittest.TestCase):

    def test_critical_risk(self):
        result = evaluate_maintenance_v2(10, 5, 15)
        self.assertEqual(result.risk_level, "CRITICAL")
        self.assertEqual(
            result.maintenance_action,
            "URGENT_MAINTENANCE_REVIEW",
        )

    def test_high_risk(self):
        result = evaluate_maintenance_v2(25, 20, 30)
        self.assertEqual(result.risk_level, "HIGH")

    def test_moderate_risk(self):
        result = evaluate_maintenance_v2(50, 40, 60)
        self.assertEqual(result.risk_level, "MODERATE")

    def test_low_risk(self):
        result = evaluate_maintenance_v2(90, 80, 100)
        self.assertEqual(result.risk_level, "LOW")

    def test_high_confidence(self):
        result = evaluate_maintenance_v2(90, 75, 100)
        self.assertEqual(result.confidence_level, "HIGH")
        self.assertFalse(result.model_review_required)

    def test_medium_confidence(self):
        result = evaluate_maintenance_v2(90, 50, 110)
        self.assertEqual(result.confidence_level, "MEDIUM")

    def test_low_confidence_triggers_review(self):
        result = evaluate_maintenance_v2(90, 0, 125)
        self.assertEqual(result.confidence_level, "LOW")
        self.assertTrue(result.model_review_required)

    def test_uncertainty_does_not_escalate_failure_risk(self):
        result = evaluate_maintenance_v2(90, 0, 125)
        self.assertEqual(result.risk_level, "LOW")
        self.assertEqual(
            result.maintenance_action,
            "CONTINUE_MONITORING",
        )

    def test_drift_detected_triggers_review(self):
        result = evaluate_maintenance_v2(
            90, 75, 100, drift_status="drift_detected"
        )
        self.assertTrue(result.model_review_required)
        self.assertEqual(result.risk_level, "LOW")

    def test_invalid_drift_status(self):
        with self.assertRaises(ValueError):
            evaluate_maintenance_v2(
                50, 40, 60, drift_status="unknown"
            )

    def test_invalid_interval(self):
        with self.assertRaises(ValueError):
            evaluate_maintenance_v2(50, 60, 40)

    def test_prediction_outside_interval(self):
        with self.assertRaises(ValueError):
            evaluate_maintenance_v2(70, 20, 60)


if __name__ == "__main__":
    unittest.main()
