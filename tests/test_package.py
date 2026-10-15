import privacy_masker
from privacy_masker import exceptions, entity_types


def test_version():
    assert privacy_masker.__version__ == "0.11.0"


def test_exception_hierarchy():
    for cls in (
        exceptions.ConfigurationError,
        exceptions.DetectorError,
        exceptions.StoreError,
        exceptions.RestoreError,
    ):
        assert issubclass(cls, exceptions.PrivacyMaskerError)
    assert issubclass(exceptions.PrivacyMaskerError, Exception)


def test_entity_type_constants():
    assert entity_types.PERSON == "PERSON"
    assert entity_types.MY_NUMBER == "MY_NUMBER"
    assert entity_types.PHONE_NUMBER == "PHONE_NUMBER"
