"""
Unit tests for production validation script.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the parent directory to the path to import production_validation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from production_validation import ValidationConfig, ProductionValidator


class TestValidationConfig:
    """Test ValidationConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ValidationConfig()
        
        assert config.seed == 42
        assert config.max_steps == 50
        assert config.output_dir == "validation_artifacts"
        assert config.batch_size == 2
        assert config.sequence_length == 1024
        assert config.vocab_size == 32000
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ValidationConfig(
            seed=123,
            max_steps=25,
            output_dir="/tmp/test",
            batch_size=4
        )
        
        assert config.seed == 123
        assert config.max_steps == 25
        assert config.output_dir == "/tmp/test"
        assert config.batch_size == 4
    
    def test_config_to_dict(self):
        """Test configuration serialization."""
        config = ValidationConfig(seed=123, max_steps=25)
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['seed'] == 123
        assert config_dict['max_steps'] == 25


class TestProductionValidator:
    """Test ProductionValidator class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def validation_config(self, temp_dir):
        """Create validation config for testing."""
        return ValidationConfig(
            output_dir=str(temp_dir / "validation_output"),
            max_steps=10,
            config_path="configs/bigbird_s5_hrm_1_2b.yaml",
            base_config_path="configs/valkyrie_base.yaml"
        )
    
    def test_validator_initialization(self, validation_config):
        """Test validator initialization."""
        validator = ProductionValidator(validation_config)
        
        assert validator.config == validation_config
        assert validator.output_dir.exists()
        assert validator.validation_results == {}
        assert validator.model is None
        assert validator.params is None
    
    def test_deterministic_environment_setup(self, validation_config):
        """Test deterministic environment setup."""
        validator = ProductionValidator(validation_config)
        
        # Check that the validator has the expected attributes
        assert hasattr(validator, 'rng')
        assert validator.config.seed == validation_config.seed
    
    def test_load_configurations_missing_files(self, validation_config):
        """Test configuration loading with missing files."""
        validator = ProductionValidator(validation_config)
        
        # This should handle missing config files gracefully
        result = validator.load_configs()
        
        # Should return True even with missing files (uses defaults)
        assert result is True
        assert 'config_loading' in validator.validation_results
        assert validator.validation_results['config_loading']['status'] == 'success'
    
    @patch('builtins.open')
    @patch('pathlib.Path.exists')
    def test_load_configurations_success(self, mock_exists, mock_open, validation_config):
        """Test successful configuration loading."""
        # Mock file existence
        mock_exists.return_value = True
        
        # Mock file content
        mock_config_content = """
        model:
          vocab_size: 32000
          d_model: 512
        training:
          learning_rate: 0.001
        """
        mock_open.return_value.__enter__.return_value.read.return_value = mock_config_content
        
        validator = ProductionValidator(validation_config)
        
        with patch('yaml.safe_load') as mock_yaml:
            mock_yaml.return_value = {'model': {'vocab_size': 32000}, 'training': {'learning_rate': 0.001}}
            result = validator.load_configs()
        
        assert result is True
        assert 'config_loading' in validator.validation_results
        assert validator.validation_results['config_loading']['status'] == 'success'


class TestValidationIntegration:
    """Integration tests for validation pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_basic_validation_flow(self, temp_dir):
        """Test basic validation flow without full execution."""
        config = ValidationConfig(
            output_dir=str(temp_dir / "validation"),
            max_steps=5,
            config_path="configs/bigbird_s5_hrm_1_2b.yaml",
            base_config_path="configs/valkyrie_base.yaml"
        )
        
        validator = ProductionValidator(config)
        
        # Test that validator can be created and basic methods work
        assert validator.config == config
        assert validator.output_dir.exists()
        
        # Test config loading (should work with missing files)
        result = validator.load_configs()
        assert result is True


class TestDeterministicExecution:
    """Test deterministic execution properties."""
    
    def test_seed_consistency(self):
        """Test that same seeds produce consistent results."""
        config1 = ValidationConfig(seed=42)
        config2 = ValidationConfig(seed=42)
        
        # Both configs should have the same seed
        assert config1.seed == config2.seed == 42
        
        # Test that different seeds produce different configs
        config3 = ValidationConfig(seed=123)
        assert config3.seed != config1.seed
    
    def test_validator_seed_consistency(self):
        """Test that validators with same seed have consistent setup."""
        config1 = ValidationConfig(seed=42)
        config2 = ValidationConfig(seed=42)
        
        with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
            config1.output_dir = str(Path(tmpdir1) / "val1")
            config2.output_dir = str(Path(tmpdir2) / "val2")
            
            validator1 = ProductionValidator(config1)
            validator2 = ProductionValidator(config2)
            
            # Both should have the same seed
            assert validator1.config.seed == validator2.config.seed == 42


if __name__ == "__main__":
    pytest.main([__file__])