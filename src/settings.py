import yaml
from pathlib import Path


class Settings:
    def __init__(self, config_path: str = "config.yaml"):
        config_file = Path(config_path)
        if not config_file.is_file():
            raise FileNotFoundError(f"Configuration file not found at '{config_path}'")

        self.PROJECT_ROOT = config_file.parent

        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        self.BOOKS_DIR = self.PROJECT_ROOT / config['data']['books_directory']
        self.CHARACTER_FILE = self.PROJECT_ROOT / config['data']['character_file']
        self.RESULTS_DIR = self.PROJECT_ROOT / config['data']['llm_results_dir']
        self.GRAPH_ARTIFACTS_DIR = self.PROJECT_ROOT / config['data']['graph_artifacts_dir']

        self.LLM_MODEL = config['models']['llm_model']
        self.LLM_HOST = config['models']['llm_host']

        self.TARGET_BOOK_FILENAME = config['processing']['target_book_filename']
        self.TOP_N_ANALYSIS = config['analysis']['top_n_results']