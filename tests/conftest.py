from dotenv import load_dotenv

from tests.test_fixtures import (  # noqa: F401
    assert_angle_close,
    assert_coordinates_close,
    geometry_test_data,
)

# Ensure environment variables from .env are available during test collection
load_dotenv()
