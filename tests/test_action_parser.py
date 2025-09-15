"""
Tests for action parsing functionality.

Tests robust action parsing with validation, error handling,
and support for multiple action formats.
"""

import pytest

from geoguess_env.action_parser import ActionParser, ActionParsingError


class TestActionParser:
    """Test cases for ActionParser class."""

    @pytest.fixture
    def parser(self):
        """Create action parser with standard image dimensions."""
        return ActionParser(image_width=1024, image_height=512)

    def test_parse_click_action_dict(self, parser):
        """Test parsing click action from dictionary."""
        action = {"op": "click", "value": [100, 200]}
        op, values = parser.parse_action(action)

        assert op == 0
        assert values == (100.0, 200.0)

    def test_parse_answer_action_dict(self, parser):
        """Test parsing answer action from dictionary."""
        action = {"op": "answer", "value": [40.7128, -74.0060]}
        op, values = parser.parse_action(action)

        assert op == 1
        assert values == (40.7128, -74.0060)

    def test_parse_click_action_string_op(self, parser):
        """Test parsing click action with string operation."""
        action = {"op": "click", "value": [512, 256]}
        op, values = parser.parse_action(action)

        assert op == 0
        assert values == (512.0, 256.0)

    def test_parse_answer_action_string_op(self, parser):
        """Test parsing answer action with string operation."""
        action = {"op": "answer", "value": [47.6062, -122.3321]}
        op, values = parser.parse_action(action)

        assert op == 1
        assert values == (47.6062, -122.3321)

    def test_parse_click_requires_value_key(self, parser):
        """Click must be provided via single 'value' key."""
        action = {"op": "click", "value": [300, 150]}
        op, values = parser.parse_action(action)
        assert op == 0
        assert values == (300.0, 150.0)

    def test_parse_answer_requires_value_key(self, parser):
        """Answer must be provided via single 'value' key."""
        action = {"op": "answer", "value": [51.5074, -0.1278]}
        op, values = parser.parse_action(action)
        assert op == 1
        assert values == (51.5074, -0.1278)

    def test_parse_json_string_click(self, parser):
        """Test parsing click action from JSON string."""
        action_str = '{"op":"click","value":[400,300]}'
        op, values = parser.parse_action(action_str)

        assert op == 0
        assert values == (400.0, 300.0)

    def test_parse_json_string_answer(self, parser):
        """Test parsing answer action from JSON string."""
        action_str = '{"op":"answer","value":[35.6762,139.6503]}'
        op, values = parser.parse_action(action_str)

        assert op == 1
        assert values == (35.6762, 139.6503)

    def test_parse_json_string_with_whitespace(self, parser):
        """Test parsing JSON string with extra whitespace."""
        action_str = '  {"op": "click", "value": [100, 200]}  '
        op, values = parser.parse_action(action_str)

        assert op == 0
        assert values == (100.0, 200.0)

    def test_click_coordinate_clamping(self, parser):
        """Test that click coordinates are clamped to image bounds."""
        # Test coordinates outside image bounds
        action = {"op": "click", "value": [2000, 1000]}
        op, values = parser.parse_action(action)

        assert op == 0
        assert values == (1023.0, 511.0)  # Clamped to max valid coordinates

        # Test negative coordinates
        action = {"op": "click", "value": [-50, -25]}
        op, values = parser.parse_action(action)

        assert op == 0
        assert values == (0.0, 0.0)  # Clamped to minimum coordinates

    def test_answer_coordinate_clamping(self, parser):
        """Test that answer coordinates are clamped to valid ranges."""
        # Test coordinates outside valid ranges
        action = {"op": "answer", "value": [95.0, 185.0]}
        op, values = parser.parse_action(action)

        assert op == 1
        assert values == (90.0, 180.0)  # Clamped to valid ranges

        # Test negative out-of-range coordinates
        action = {"op": "answer", "value": [-95.0, -185.0]}
        op, values = parser.parse_action(action)

        assert op == 1
        assert values == (-90.0, -180.0)

    def test_parse_with_fallback_success(self, parser):
        """Test parse_with_fallback with valid action."""
        action = {"op": "click", "value": [100, 200]}
        op, values = parser.parse_with_fallback(action)

        assert op == 0
        assert values == (100.0, 200.0)

    def test_parse_with_fallback_invalid_action(self, parser):
        """Test parse_with_fallback with invalid action falls back to center click."""
        action = {"invalid": "action"}
        op, values = parser.parse_with_fallback(action)

        assert op == 0
        assert values == (512.0, 256.0)  # Center of 1024x512 image

    def test_parse_with_fallback_invalid_json(self, parser):
        """Test parse_with_fallback with invalid JSON falls back to center click."""
        action_str = '{"invalid json'
        op, values = parser.parse_with_fallback(action_str)

        assert op == 0
        assert values == (512.0, 256.0)

    def test_missing_op_field(self, parser):
        """Test error when 'op' field is missing."""
        action = {"value": [100, 200]}
        with pytest.raises(ActionParsingError, match="Action missing 'op' field"):
            parser.parse_action(action)

    def test_invalid_operation(self, parser):
        """Test error with invalid operation."""
        action = {"op": "invalid_op", "value": [100, 200]}
        with pytest.raises(ActionParsingError, match="Invalid operation"):
            parser.parse_action(action)

    def test_missing_click_values(self, parser):
        """Test error when click values are missing (no 'value')."""
        action = {"op": "click"}
        with pytest.raises(ActionParsingError, match="Action missing 'value' field"):
            parser.parse_action(action)

    def test_missing_answer_values(self, parser):
        """Test error when answer values are missing (no 'value')."""
        action = {"op": "answer"}
        with pytest.raises(ActionParsingError, match="Action missing 'value' field"):
            parser.parse_action(action)

    def test_invalid_click_values_format(self, parser):
        """Test error with invalid click values format."""
        action = {"op": "click", "value": [100]}  # Only one coordinate
        with pytest.raises(
            ActionParsingError, match="Click values must be \\[x, y\\] array"
        ):
            parser.parse_action(action)

    def test_invalid_answer_values_format(self, parser):
        """Test error with invalid answer values format."""
        action = {"op": "answer", "value": "not_a_list"}
        with pytest.raises(
            ActionParsingError, match="Answer values must be \\[lat, lon\\] array"
        ):
            parser.parse_action(action)

    def test_non_numeric_click_coordinates(self, parser):
        """Test error with non-numeric click coordinates."""
        action = {"op": "click", "value": ["not", "numbers"]}
        with pytest.raises(
            ActionParsingError, match="Click coordinates must be numeric"
        ):
            parser.parse_action(action)

    def test_non_numeric_answer_coordinates(self, parser):
        """Test error with non-numeric answer coordinates."""
        action = {"op": "answer", "value": ["not", "numbers"]}
        with pytest.raises(
            ActionParsingError, match="Answer coordinates must be numeric"
        ):
            parser.parse_action(action)

    def test_invalid_json_string(self, parser):
        """Test error with invalid JSON string."""
        action_str = '{"invalid": json}'
        with pytest.raises(ActionParsingError, match="Invalid JSON string"):
            parser.parse_action(action_str)

    def test_non_dict_action(self, parser):
        """Test error with non-dictionary action."""
        action = "not a dict"
        with pytest.raises(
            ActionParsingError, match="Action must be dict or JSON string"
        ):
            parser.parse_action(action)

    def test_validate_click_action(self, parser):
        """Test click action validation."""
        # Valid coordinates
        assert parser.validate_click_action(100, 200)
        assert parser.validate_click_action(0, 0)
        assert parser.validate_click_action(1023, 511)

        # Invalid coordinates
        assert not parser.validate_click_action(-1, 200)
        assert not parser.validate_click_action(1024, 200)
        assert not parser.validate_click_action(100, -1)
        assert not parser.validate_click_action(100, 512)

    def test_validate_answer_action(self, parser):
        """Test answer action validation."""
        # Valid coordinates
        assert parser.validate_answer_action(40.7128, -74.0060)
        assert parser.validate_answer_action(0, 0)
        assert parser.validate_answer_action(90, 180)
        assert parser.validate_answer_action(-90, -180)

        # Invalid coordinates
        assert not parser.validate_answer_action(91, 0)
        assert not parser.validate_answer_action(-91, 0)
        assert not parser.validate_answer_action(0, 181)
        assert not parser.validate_answer_action(0, -181)

    def test_create_click_action(self, parser):
        """Test creating properly formatted click action."""
        action = parser.create_click_action(100, 200)
        assert action["op"] == "click"
        assert action["value"] == [100, 200]
        assert set(action.keys()) == {"op", "value"}

    def test_create_click_action_with_clamping(self, parser):
        """Test creating click action with coordinate clamping."""
        action = parser.create_click_action(2000, 1000)
        assert action["value"] == [1023, 511]  # Clamped values

    def test_create_answer_action(self, parser):
        """Test creating properly formatted answer action."""
        action = parser.create_answer_action(40.7128, -74.0060)
        assert action["op"] == "answer"
        assert action["value"] == [40.7128, -74.0060]
        assert set(action.keys()) == {"op", "value"}

    def test_create_answer_action_with_clamping(self, parser):
        """Test creating answer action with coordinate clamping."""
        action = parser.create_answer_action(95.0, 185.0)
        assert action["value"] == [90.0, 180.0]  # Clamped values

    def test_get_center_click(self, parser):
        """Test getting center click action."""
        action = parser.get_center_click()
        assert action["op"] == "click"
        assert action["value"] == [512, 256]  # Center of 1024x512 image

    def test_invalid_operation_type(self, parser):
        """Operation must be a valid string token."""
        action = {"op": "invalid", "value": [100, 200]}
        with pytest.raises(ActionParsingError, match="Invalid operation"):
            parser.parse_action(action)
