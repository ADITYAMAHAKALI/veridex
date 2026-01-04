import unittest
from unittest.mock import patch, MagicMock
import os
from veridex.video.weights import get_weight_config, set_weight_url, DEFAULT_WEIGHTS

class TestVideoWeights(unittest.TestCase):
    
    def test_get_weight_config_defaults(self):
        # Test defaults
        config = get_weight_config('physnet')
        self.assertEqual(config['url'], DEFAULT_WEIGHTS['physnet']['url'])
        self.assertEqual(config['filename'], 'physnet.pth')
        
    def test_get_weight_config_unknown(self):
        with self.assertRaises(ValueError):
            get_weight_config('unknown_model')
            
    def test_get_weight_config_env_var(self):
        # Test env override
        with patch.dict(os.environ, {'VERIDEX_PHYSNET_URL': 'http://test.com/model.pth'}):
            config = get_weight_config('physnet')
            self.assertEqual(config['url'], 'http://test.com/model.pth')
            
    def test_set_weight_url(self):
        # Save original to restore later (though patched dict might be safer if I could patch the module dict)
        original_url = DEFAULT_WEIGHTS['physnet']['url']
        
        try:
            # Set new url
            set_weight_url('physnet', 'http://custom.url/model.pth', sha256='abc')
            
            config = get_weight_config('physnet')
            self.assertEqual(config['url'], 'http://custom.url/model.pth')
            self.assertEqual(config['sha256'], 'abc')
            
            # Test unknown model
            with self.assertRaises(ValueError):
                set_weight_url('unknown', 'http://url')
                
        finally:
            # Restore
            DEFAULT_WEIGHTS['physnet']['url'] = original_url
            DEFAULT_WEIGHTS['physnet']['sha256'] = None

if __name__ == '__main__':
    unittest.main()
