import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Add the parent directory to sys.path to import main
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import main

class TestModelDownloader(unittest.TestCase):

    @patch('main.snapshot_download')
    @patch('argparse.ArgumentParser.parse_args')
    def test_main_default_args(self, mock_parse_args, mock_snapshot_download):
        # Setup mock arguments
        mock_args = MagicMock()
        mock_args.model = "openai/whisper-small"
        mock_parse_args.return_value = mock_args

        main.main()

        # Check if snapshot_download was called with expected arguments
        mock_snapshot_download.assert_called_once_with(
            repo_id="openai/whisper-small",
            allow_patterns=["*.json", "*.safetensors"],
            local_dir="../../whisper-small",
        )

    @patch('main.snapshot_download')
    @patch('argparse.ArgumentParser.parse_args')
    def test_main_custom_model(self, mock_parse_args, mock_snapshot_download):
        # Setup mock arguments
        mock_args = MagicMock()
        mock_args.model = "openai/whisper-tiny"
        mock_parse_args.return_value = mock_args

        main.main()

        # Check if snapshot_download was called with custom model and derived path
        mock_snapshot_download.assert_called_once_with(
            repo_id="openai/whisper-tiny",
            allow_patterns=["*.json", "*.safetensors"],
            local_dir="../../whisper-tiny",
        )

if __name__ == '__main__':
    unittest.main()
