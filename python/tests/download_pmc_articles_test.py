import os
import unittest
from unittest.mock import patch, MagicMock
from Bio import Entrez
import time
from modules import Downloader

class TestDownloadPmcArticles(unittest.TestCase):
    @patch('Refactor.NCBIManager.fetch_pmc_ids')
    @patch('Refactor.Entrez.efetch')
    @patch('os.makedirs')
    def test_no_articles_found(self, mock_makedirs, mock_efetch, mock_fetch_pmc_ids):
        mock_fetch_pmc_ids.return_value = []
        
        with patch('builtins.print') as mocked_print:
            download_pmc_articles('dummy_query', 'dummy_dir')
            mocked_print.assert_called_with('No articles found for query: dummy_query')
    
    @patch('Refactor.NCBIManager.fetch_pmc_ids')
    @patch('Refactor.Entrez.efetch')
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('os.makedirs')
    def test_successful_download(self, mock_makedirs, mock_open, mock_efetch, mock_fetch_pmc_ids):
        mock_fetch_pmc_ids.return_value = ['PMC12345']
        mock_efetch.return_value.read.return_value = b"<xml>article content</xml>"
        
        with patch('builtins.print') as mocked_print:
            download_pmc_articles('dummy_query', 'dummy_dir')
            mocked_print.assert_any_call('Article PMC12345 downloaded.')
    
    @patch('Refactor.NCBIManager.fetch_pmc_ids')
    @patch('Refactor.Entrez.efetch')
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('os.makedirs')
    def test_retry_on_failure(self, mock_makedirs, mock_open, mock_efetch, mock_fetch_pmc_ids):
        mock_fetch_pmc_ids.return_value = ['PMC12345']
        mock_efetch.side_effect = [Exception("Network error"), MagicMock(read=MagicMock(return_value=b"<xml>article content</xml>"))]
        
        with patch('builtins.print') as mocked_print:
            download_pmc_articles('dummy_query', 'dummy_dir', max_retries=2)
            mocked_print.assert_any_call('Error downloading article PMC12345: Network error. Retrying...')
            mocked_print.assert_any_call('Article PMC12345 downloaded.')
    
    @patch('Refactor.NCBIManager.fetch_pmc_ids')
    @patch('Refactor.Entrez.efetch')
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('os.makedirs')
    @patch('time.sleep', return_value=None)
    def test_timeout_restart(self, mock_sleep, mock_makedirs, mock_open, mock_efetch, mock_fetch_pmc_ids):
        mock_fetch_pmc_ids.return_value = ['PMC12345', 'PMC67890']
        mock_efetch.side_effect = [MagicMock(read=MagicMock(return_value=b"<xml>article content</xml>")),
                                   Exception("Network error")]
        
        with patch('builtins.print') as mocked_print:
            with patch('time.time', side_effect=[0, 1, 35, 36]):  # Simulate timeout after the first download
                download_pmc_articles('dummy_query', 'dummy_dir', max_retries=1, timeout=30)
                mocked_print.assert_any_call('Timeout exceeded. Restarting from article PMC67890.')
                # Ensure the function was called twice
                self.assertEqual(mocked_print.call_count, 6)
    
if __name__ == '__main__':
    unittest.main()
