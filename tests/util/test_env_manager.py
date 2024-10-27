import os
from collections.abc import Generator
from pathlib import Path

import pytest
from pydantic import SecretStr

from llm_try.util.env_manager import (  # 実際のモジュール名に置き換えてください
    EnvManager,
    get_anthropic_api_key,
    get_anthropic_api_key_required,
    get_api_key,
    get_api_key_required,
    get_openai_api_key,
    get_openai_api_key_required,
)


@pytest.fixture
def mock_env_file(tmp_path: Path) -> Generator[Path, None, None]:
    """一時的な.envファイルを作成するフィクスチャ"""
    env_file = tmp_path / ".env"
    env_file.write_text("")  # 空の.envファイルを作成
    original_cwd = Path.cwd()

    # テストの作業ディレクトリを一時ディレクトリに変更
    os.chdir(tmp_path)
    yield env_file
    # 作業ディレクトリを元に戻す
    os.chdir(original_cwd)


@pytest.fixture
def mock_env_vars(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """環境変数をモックするフィクスチャ"""
    # テスト用の環境変数を設定
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    yield
    # テスト後にクリーンアップ
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture
def clear_env_vars(monkeypatch: pytest.MonkeyPatch, mock_env_file: Path) -> None:  # noqa: ARG001   # noqa: F811
    """環境変数をクリアするフィクスチャ"""
    # 既存の環境変数を一時的に削除
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


class TestEnvManager:
    @pytest.mark.usefixtures("clear_env_vars")
    def test_env_manager_initialization(self: None) -> None:
        """EnvManagerの初期化テスト"""
        manager = EnvManager()
        assert isinstance(manager.anthropic_api_key, SecretStr | type(None))
        assert isinstance(manager.openai_api_key, SecretStr | type(None))

    @pytest.mark.usefixtures("mock_env_vars")
    def test_env_manager_with_env_vars(self) -> None:
        """環境変数が設定されている場合のテスト"""
        manager = EnvManager()
        assert manager.anthropic_api_key is not None
        assert manager.openai_api_key is not None
        assert manager.anthropic_api_key.get_secret_value() == "test-anthropic-key"
        assert manager.openai_api_key.get_secret_value() == "test-openai-key"

    @pytest.mark.usefixtures("clear_env_vars")
    def test_env_manager_without_env_vars(self) -> None:
        """環境変数が設定されていない場合のテスト"""
        manager = EnvManager()
        assert manager.anthropic_api_key is None
        assert manager.openai_api_key is None


class TestGetApiKey:
    @pytest.mark.usefixtures("mock_env_vars")
    def test_get_api_key_with_env_vars(self) -> None:
        """環境変数が設定されている場合のget_api_keyのテスト"""
        assert get_api_key("ANTHROPIC_API_KEY") == "test-anthropic-key"
        assert get_api_key("OPENAI_API_KEY") == "test-openai-key"

    @pytest.mark.usefixtures("clear_env_vars")
    def test_get_api_key_without_env_vars(self) -> None:
        """環境変数が設定されていない場合のget_api_keyのテスト"""
        assert get_api_key("ANTHROPIC_API_KEY") is None
        assert get_api_key("OPENAI_API_KEY") is None

    @pytest.mark.usefixtures("mock_env_vars")
    def test_get_api_key_with_invalid_key(self) -> None:
        """無効なキー名でのget_api_keyのテスト"""
        assert get_api_key("INVALID_KEY") is None


class TestGetApiKeyRequired:
    @pytest.mark.usefixtures("mock_env_vars")
    def test_get_api_key_required_with_env_vars(self) -> None:
        """環境変数が設定されている場合のget_api_key_requiredのテスト"""
        assert get_api_key_required("ANTHROPIC_API_KEY") == "test-anthropic-key"
        assert get_api_key_required("OPENAI_API_KEY") == "test-openai-key"

    @pytest.mark.usefixtures("clear_env_vars")
    def test_get_api_key_required_without_env_vars(self) -> None:
        """環境変数が設定されていない場合のget_api_key_requiredのテスト"""
        with pytest.raises(
            ValueError,
            match="ANTHROPIC_API_KEY is not set in environment variables or .env file",
        ):
            get_api_key_required("ANTHROPIC_API_KEY")


class TestPartialFunctions:
    @pytest.mark.usefixtures("mock_env_vars")
    def test_service_specific_functions_with_env_vars(self) -> None:
        """サービス固有の関数のテスト(環境変数あり)"""
        assert get_anthropic_api_key() == "test-anthropic-key"
        assert get_openai_api_key() == "test-openai-key"
        assert get_anthropic_api_key_required() == "test-anthropic-key"
        assert get_openai_api_key_required() == "test-openai-key"

    @pytest.mark.usefixtures("clear_env_vars")
    def test_service_specific_functions_without_env_vars(self) -> None:
        """サービス固有の関数のテスト(環境変数なし)"""
        assert get_anthropic_api_key() is None
        assert get_openai_api_key() is None

        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
            get_anthropic_api_key_required()

        with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
            get_openai_api_key_required()


@pytest.fixture(autouse=True)
def cleanup_env_after_test(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[None, None, None]:
    """
    各テスト後に環境変数をクリーンアップするフィクスチャ
    このフィクスチャは自動的に全てのテストに適用されます
    """
    yield
    # テスト後に環境変数を元の状態に戻す
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
