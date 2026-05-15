from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parents[1]


def helper_path(*parts: str, trailing_slash: bool = False) -> str:
    path = TESTS_DIR.joinpath("helpers", *parts)
    path_str = str(path)
    return f"{path_str}/" if trailing_slash else path_str


def device_parameters_path(device_name: str) -> str:
    return helper_path("device_parameters", device_name, trailing_slash=True)
