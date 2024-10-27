from functools import partial

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvManager(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    anthropic_api_key: SecretStr | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    openai_api_key: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY")


def get_api_key(key_name: str = "ANTHROPIC_API_KEY") -> str | None:
    """
    指定されたキー名に対応する環境変数の値を取得します

    Args:
        key_name (str): 取得したい環境変数のキー名

    Returns:
        Optional[str]: 環境変数の値。設定されていない場合はNone
    """
    env_manager = EnvManager()
    env_dict = env_manager.model_dump()

    key_snake_case = key_name.lower()

    if key_snake_case not in env_dict:
        return None

    value = env_dict[key_snake_case]

    if isinstance(value, SecretStr):
        return value.get_secret_value()

    return value if isinstance(value, str | type(None)) else None


def get_api_key_required(key_name: str = "ANTHROPIC_API_KEY") -> str:
    """
    指定されたキー名に対応する環境変数の値を取得します(必須バージョン)

    Args:
        key_name (str): 取得したい環境変数のキー名

    Returns:
        str: 環境変数の値

    Raises:
        ValueError: 指定されたキーが設定されていない場合
    """
    value = get_api_key(key_name)
    if value is None:
        msg = f"{key_name} is not set in environment variables or .env file"
        raise ValueError(msg)
    return value


# 各サービス用のAPI取得関数を partial を使って作成
get_anthropic_api_key = partial(get_api_key, key_name="ANTHROPIC_API_KEY")
get_openai_api_key = partial(get_api_key, key_name="OPENAI_API_KEY")

# 必須バージョンも同様に作成
get_anthropic_api_key_required = partial(
    get_api_key_required, key_name="ANTHROPIC_API_KEY"
)
get_openai_api_key_required = partial(get_api_key_required, key_name="OPENAI_API_KEY")
