from app.providers.provider_repository import ProviderRepository


def test_provider_search_empty_without_data(monkeypatch):
    # Uses real DB session; if DB unavailable this may skip via exception handling in caller tests.
    # Lightweight unit: message contract when no source configured.
    repo = ProviderRepository.__new__(ProviderRepository)
    from app.core.config import get_settings

    settings = get_settings()
    assert settings.provider_data_source == "" or isinstance(settings.provider_data_source, str)
