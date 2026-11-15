import unittest

from src.decision.maintenance_engine import evaluate_maintenance


class TestMaintenanceEngine(unittest.TestCase):

    def test_critical_rul(self):
        result = evaluate_maintenance(10, 5, 15)
        self.assertEqual(result.priority, "URGENT")

    def test_schedule_soon(self):
        result = evaluate_maintenance(25, 20, 30)
        self.assertEqual(result.priority, "SCHEDULE_SOON")

    def test_plan_maintenance(self):
        result = evaluate_maintenance(50, 40, 60)
        self.assertEqual(result.priority, "PLAN")

    def test_monitor(self):
        result = evaluate_maintenance(90, 80, 100)
        self.assertEqual(result.priority, "MONITOR")

    def test_wide_interval_increases_priority(self):
        result = evaluate_maintenance(90, 65, 110)
        self.assertEqual(result.priority, "PLAN")

    def test_drift_warning_increases_priority(self):
        result = evaluate_maintenance(
            90, 80, 100, drift_status="warning"
        )
        self.assertEqual(result.priority, "PLAN")

    def test_detected_drift_increases_priority(self):
        result = evaluate_maintenance(
            90, 80, 100, drift_status="drift_detected"
        )
        self.assertEqual(result.priority, "SCHEDULE_SOON")

    def test_negative_lower_bound_is_treated_as_zero(self):
        result = evaluate_maintenance(5, -5, 15)
        self.assertEqual(result.effective_lower_bound, 0)
        self.assertEqual(result.priority, "URGENT")

    def test_invalid_interval_raises_error(self):
        with self.assertRaises(ValueError):
            evaluate_maintenance(50, 60, 40)

    def test_invalid_drift_status_raises_error(self):
        with self.assertRaises(ValueError):
            evaluate_maintenance(
                50, 40, 60, drift_status="unknown"
            )


if __name__ == "__main__":
    unittest.main()