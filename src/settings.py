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

        # Paths
        self.BOOKS_DIR = self.PROJECT_ROOT / config['data']['books_directory']
        self.CHARACTER_FILE = self.PROJECT_ROOT / config['data']['character_file']
        self.RESULTS_DIR = self.PROJECT_ROOT / config['data']['llm_results_dir']
        self.GRAPH_ARTIFACTS_DIR = self.PROJECT_ROOT / config['data']['graph_artifacts_dir']

        # Models
        self.LLM_MODEL = config['models']['llm_model']
        self.LLM_HOST = config['models']['llm_host']
        self.FAST_TOKENIZER = config['models']['fast_tokenizer_for_counting']

        # Processing
        # THE CHANGE: Load token-based chunking settings
        self.CHUNK_TOKEN_LIMIT = config['processing']['chunk_token_limit']
        self.CHUNK_OVERLAP_SENTENCES = config['processing']['chunk_overlap_sentences']

        # Analysis
        self.TOP_N_ANALYSIS = config['analysis']['top_n_results']