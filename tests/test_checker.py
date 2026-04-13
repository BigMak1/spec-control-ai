from backend.app.tools import compare_values


class TestCompareValues:
    def test_gte_pass(self):
        result = compare_values("4.0", "2.5", "gte")
        assert result["match"] is True
        assert result["actual_parsed"] == 4.0
        assert result["required_parsed"] == 2.5

    def test_gte_fail(self):
        result = compare_values("1.5", "2.5", "gte")
        assert result["match"] is False

    def test_lte_pass(self):
        result = compare_values("10", "15", "lte")
        assert result["match"] is True

    def test_lte_fail(self):
        result = compare_values("20", "15", "lte")
        assert result["match"] is False

    def test_eq_pass(self):
        result = compare_values("380", "380", "eq")
        assert result["match"] is True

    def test_eq_fail(self):
        result = compare_values("220", "380", "eq")
        assert result["match"] is False

    def test_contains_pass(self):
        result = compare_values("ВВГнг(А)-LS", "ВВГнг", "contains")
        assert result["match"] is True

    def test_contains_fail(self):
        result = compare_values("АВВГнг(А)-LSLTx", "ВВГнг", "contains")
        assert result["match"] is False

    def test_unparseable_numbers(self):
        result = compare_values("медные жилы", "2.5", "gte")
        assert result["match"] is None
        assert "cannot parse" in result["explanation"]

    def test_units_in_value(self):
        result = compare_values("2.5 мм²", "2.5", "gte")
        assert result["match"] is True
        assert result["actual_parsed"] == 2.5
