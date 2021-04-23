from .context import interfaces


def test_interfaces_constructor():
    try:
        interfaces.RegistrationInterface(name="Test")
    except TypeError:
        pass
