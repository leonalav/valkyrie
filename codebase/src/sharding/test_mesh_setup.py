"""
Comprehensive unit tests for enhanced TPU mesh setup functionality.

Tests cover:
- Basic mesh creation with various device counts
- Environment variable configuration
- Global mesh storage and thread safety
- Topology assertions and experimental utils
- Error handling and edge cases
- Multi-host and single-host scenarios
"""

import os
import pytest
import threading
import time
from unittest.mock import Mock, patch, MagicMock
from typing import List, Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

# Import the module under test
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sharding.mesh_setup import (
    get_tpu_config_from_env,
    get_optimal_topology,
    create_device_mesh_with_assertions,
    make_mesh,
    get_global_mesh,
    set_global_mesh,
    clear_global_mesh,
    make_or_get_global_mesh,
    validate_mesh_setup,
    get_mesh_info,
    setup_tpu_mesh,
    print_env_usage
)


class TestEnvironmentConfiguration:
    """Test environment variable configuration parsing."""
    
    def setup_method(self):
        """Clear environment variables before each test."""
        env_vars = [
            'TPU_MESH_CONFIG', 'TPU_AXIS_NAMES', 'TPU_DEVICE_COUNT',
            'TPU_FORCE_TOPOLOGY', 'TPU_USE_GLOBAL_MESH'
        ]
        for var in env_vars:
            if var in os.environ:
                del os.environ[var]
    
    def test_empty_environment(self):
        """Test configuration with no environment variables set."""
        config = get_tpu_config_from_env()
        
        assert config['device_count'] is None
        assert config['topology'] is None
        assert config['axis_names'] is None
        assert config['force_topology'] is False
        assert config['use_global_mesh'] is True  # Default
    
    def test_mesh_config_parsing(self):
        """Test TPU_MESH_CONFIG parsing."""
        os.environ['TPU_MESH_CONFIG'] = '4,4,2'
        config = get_tpu_config_from_env()
        
        assert config['topology'] == (4, 4, 2)
        assert config['device_count'] == 32
    
    def test_axis_names_parsing(self):
        """Test TPU_AXIS_NAMES parsing."""
        os.environ['TPU_AXIS_NAMES'] = 'x,y,z'
        config = get_tpu_config_from_env()
        
        assert config['axis_names'] == ('x', 'y', 'z')
    
    def test_device_count_override(self):
        """Test TPU_DEVICE_COUNT override."""
        os.environ['TPU_DEVICE_COUNT'] = '16'
        config = get_tpu_config_from_env()
        
        assert config['device_count'] == 16
    
    def test_force_topology_flag(self):
        """Test TPU_FORCE_TOPOLOGY flag."""
        os.environ['TPU_FORCE_TOPOLOGY'] = 'true'
        config = get_tpu_config_from_env()
        
        assert config['force_topology'] is True
    
    def test_global_mesh_flag(self):
        """Test TPU_USE_GLOBAL_MESH flag."""
        os.environ['TPU_USE_GLOBAL_MESH'] = 'false'
        config = get_tpu_config_from_env()
        
        assert config['use_global_mesh'] is False
    
    def test_invalid_mesh_config(self):
        """Test handling of invalid mesh configuration."""
        os.environ['TPU_MESH_CONFIG'] = 'invalid,config'
        
        with patch('sharding.mesh_setup.logger') as mock_logger:
            config = get_tpu_config_from_env()
            mock_logger.warning.assert_called_once()
            assert config['topology'] is None
    
    def test_invalid_device_count(self):
        """Test handling of invalid device count."""
        os.environ['TPU_DEVICE_COUNT'] = 'not_a_number'
        
        with patch('sharding.mesh_setup.logger') as mock_logger:
            config = get_tpu_config_from_env()
            mock_logger.warning.assert_called_once()
            assert config['device_count'] is None


class TestOptimalTopology:
    """Test optimal topology calculation."""
    
    def test_supported_device_counts(self):
        """Test topology for supported device counts."""
        test_cases = [
            (1, ((1,), ('x',))),
            (4, ((4,), ('x',))),
            (8, ((2, 4), ('x', 'y'))),
            (16, ((4, 4), ('x', 'y'))),
            (32, ((4, 4, 2), ('x', 'y', 'z'))),
            (64, ((4, 4, 4), ('x', 'y', 'z'))),
            (128, ((8, 4, 4), ('x', 'y', 'z'))),
            (256, ((8, 8, 4), ('x', 'y', 'z'))),
        ]
        
        for device_count, expected in test_cases:
            topology, axis_names = get_optimal_topology(device_count)
            assert topology == expected[0]
            assert axis_names == expected[1]
    
    def test_factorizable_device_counts(self):
        """Test topology for factorizable device counts."""
        # Test 12 devices (2 * 2 * 3 -> not supported, but 4 * 2 * 1 should work)
        with pytest.raises(ValueError):
            get_optimal_topology(12)
        
        # Test 24 devices (2 * 4 * 3 -> not supported)
        with pytest.raises(ValueError):
            get_optimal_topology(24)
    
    def test_unsupported_device_count(self):
        """Test error for unsupported device counts."""
        with pytest.raises(ValueError, match="Unsupported device count"):
            get_optimal_topology(7)  # Prime number
        
        with pytest.raises(ValueError, match="Unsupported device count"):
            get_optimal_topology(100)  # Too many factors


class TestDeviceMeshCreation:
    """Test device mesh creation with assertions."""
    
    def create_mock_devices(self, count: int) -> List[Mock]:
        """Create mock JAX devices."""
        devices = []
        for i in range(count):
            device = Mock()
            device.platform = 'tpu'
            device.id = i
            device.process_index = 0
            devices.append(device)
        return devices
    
    def test_valid_mesh_creation(self):
        """Test successful mesh creation."""
        devices = self.create_mock_devices(8)
        topology = (2, 4)
        axis_names = ('x', 'y')
        
        with patch('numpy.array') as mock_array, \
             patch('jax.sharding.Mesh') as mock_mesh_class:
            
            mock_array.return_value.reshape.return_value = "reshaped_devices"
            mock_mesh = Mock()
            mock_mesh_class.return_value = mock_mesh
            
            result = create_device_mesh_with_assertions(devices, topology, axis_names)
            
            assert result == mock_mesh
            mock_mesh_class.assert_called_once_with("reshaped_devices", axis_names)
    
    def test_device_count_mismatch(self):
        """Test error when device count doesn't match topology."""
        devices = self.create_mock_devices(6)
        topology = (2, 4)  # Expects 8 devices
        axis_names = ('x', 'y')
        
        with pytest.raises(ValueError, match="Device count mismatch"):
            create_device_mesh_with_assertions(devices, topology, axis_names)
    
    def test_axis_names_mismatch(self):
        """Test error when axis names don't match topology dimensions."""
        devices = self.create_mock_devices(8)
        topology = (2, 4)
        axis_names = ('x', 'y', 'z')  # Too many axis names
        
        with pytest.raises(ValueError, match="Axis names count"):
            create_device_mesh_with_assertions(devices, topology, axis_names)
    
    def test_experimental_mesh_utils(self):
        """Test using experimental mesh utils when available."""
        devices = self.create_mock_devices(8)
        topology = (2, 4)
        axis_names = ('x', 'y')
        
        mock_mesh_utils = Mock()
        mock_mesh_utils.create_device_mesh.return_value = "experimental_mesh"
        mock_mesh = Mock()
        
        with patch.dict('sys.modules', {'jax.experimental.mesh_utils': mock_mesh_utils}), \
             patch('jax.sharding.Mesh', return_value=mock_mesh) as mock_mesh_class:
            
            result = create_device_mesh_with_assertions(devices, topology, axis_names)
            
            mock_mesh_utils.create_device_mesh.assert_called_once_with(topology, devices)
            mock_mesh_class.assert_called_once_with("experimental_mesh", axis_names)
            assert result == mock_mesh
    
    def test_fallback_to_1d_mesh(self):
        """Test fallback to 1D mesh on creation failure."""
        devices = self.create_mock_devices(8)
        topology = (2, 4)
        axis_names = ('x', 'y')
        
        with patch('numpy.array') as mock_array, \
             patch('jax.sharding.Mesh') as mock_mesh_class:
            
            # First call (reshape) raises exception
            mock_array.return_value.reshape.side_effect = Exception("Reshape failed")
            # Second call (1D fallback) succeeds
            mock_array.side_effect = [mock_array.return_value, devices]
            
            mock_mesh = Mock()
            mock_mesh_class.return_value = mock_mesh
            
            result = create_device_mesh_with_assertions(devices, topology, axis_names)
            
            # Should create 1D mesh as fallback
            assert mock_mesh_class.call_count == 1
            mock_mesh_class.assert_called_with(devices, ('devices',))


class TestMeshCreation:
    """Test main mesh creation function."""
    
    def setup_method(self):
        """Clear global mesh before each test."""
        clear_global_mesh()
    
    @patch('jax.devices')
    def test_basic_mesh_creation(self, mock_devices):
        """Test basic mesh creation without environment config."""
        mock_device_list = [Mock(platform='tpu') for _ in range(8)]
        mock_devices.return_value = mock_device_list
        
        with patch('sharding.mesh_setup.create_device_mesh_with_assertions') as mock_create:
            mock_mesh = Mock()
            mock_mesh.devices = mock_device_list
            mock_mesh.axis_names = ('x', 'y')
            mock_mesh.__len__ = Mock(return_value=8)
            mock_create.return_value = mock_mesh
            
            result = make_mesh(device_count=8, use_env_config=False)
            
            mock_create.assert_called_once()
            assert result == mock_mesh
    
    @patch('jax.devices')
    def test_no_devices_available(self, mock_devices):
        """Test error when no devices are available."""
        mock_devices.return_value = []
        
        with pytest.raises(RuntimeError, match="No JAX devices available"):
            make_mesh()
    
    @patch('jax.devices')
    def test_insufficient_devices(self, mock_devices):
        """Test error when requesting more devices than available."""
        mock_devices.return_value = [Mock() for _ in range(4)]
        
        with pytest.raises(ValueError, match="Requested 8 devices"):
            make_mesh(device_count=8)
    
    @patch('jax.devices')
    def test_device_subset_usage(self, mock_devices):
        """Test using subset of available devices."""
        mock_devices.return_value = [Mock(platform='tpu') for _ in range(16)]
        
        with patch('sharding.mesh_setup.create_device_mesh_with_assertions') as mock_create:
            mock_mesh = Mock()
            mock_create.return_value = mock_mesh
            
            make_mesh(device_count=8)
            
            # Should use only first 8 devices
            args, kwargs = mock_create.call_args
            devices_used = args[0]
            assert len(devices_used) == 8
    
    @patch('jax.devices')
    @patch.dict(os.environ, {'TPU_MESH_CONFIG': '4,4', 'TPU_AXIS_NAMES': 'x,y'})
    def test_environment_config_usage(self, mock_devices):
        """Test using environment configuration."""
        mock_devices.return_value = [Mock(platform='tpu') for _ in range(16)]
        
        with patch('sharding.mesh_setup.create_device_mesh_with_assertions') as mock_create:
            mock_mesh = Mock()
            mock_create.return_value = mock_mesh
            
            make_mesh()
            
            args, kwargs = mock_create.call_args
            devices, topology, axis_names = args
            assert topology == (4, 4)
            assert axis_names == ('x', 'y')


class TestGlobalMeshStorage:
    """Test global mesh storage functionality."""
    
    def setup_method(self):
        """Clear global mesh before each test."""
        clear_global_mesh()
    
    def test_global_mesh_storage(self):
        """Test storing and retrieving global mesh."""
        mock_mesh = Mock()
        config = {'test': 'config'}
        
        # Initially no global mesh
        assert get_global_mesh() is None
        
        # Set global mesh
        set_global_mesh(mock_mesh, config)
        assert get_global_mesh() == mock_mesh
        
        # Clear global mesh
        clear_global_mesh()
        assert get_global_mesh() is None
    
    def test_thread_safety(self):
        """Test thread-safe access to global mesh."""
        results = []
        errors = []
        
        def worker(mesh_id):
            try:
                mock_mesh = Mock()
                mock_mesh.id = mesh_id
                set_global_mesh(mock_mesh)
                time.sleep(0.01)  # Small delay to encourage race conditions
                retrieved = get_global_mesh()
                results.append((mesh_id, retrieved.id))
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Should have no errors
        assert len(errors) == 0
        # Should have results from all threads
        assert len(results) == 10
    
    @patch('jax.devices')
    def test_make_or_get_global_mesh_reuse(self, mock_devices):
        """Test reusing global mesh when available."""
        mock_devices.return_value = [Mock(platform='tpu') for _ in range(8)]
        mock_mesh = Mock()
        
        # Set global mesh
        set_global_mesh(mock_mesh)
        
        with patch.dict(os.environ, {'TPU_USE_GLOBAL_MESH': 'true'}):
            result = make_or_get_global_mesh()
            assert result == mock_mesh
    
    @patch('jax.devices')
    def test_make_or_get_global_mesh_create_new(self, mock_devices):
        """Test creating new mesh when global mesh not available."""
        mock_devices.return_value = [Mock(platform='tpu') for _ in range(8)]
        
        with patch('sharding.mesh_setup.make_mesh') as mock_make_mesh:
            mock_mesh = Mock()
            mock_make_mesh.return_value = mock_mesh
            
            with patch.dict(os.environ, {'TPU_USE_GLOBAL_MESH': 'true'}):
                result = make_or_get_global_mesh()
                
                mock_make_mesh.assert_called_once()
                assert result == mock_mesh
                # Should also store globally
                assert get_global_mesh() == mock_mesh


class TestMeshValidation:
    """Test mesh validation functionality."""
    
    def test_valid_mesh(self):
        """Test validation of valid mesh."""
        mock_mesh = Mock(spec=Mesh)
        mock_mesh.devices.flatten.return_value = [
            Mock(platform='tpu', process_index=0),
            Mock(platform='tpu', process_index=0)
        ]
        mock_mesh.axis_names = ('x', 'y')
        
        assert validate_mesh_setup(mock_mesh) is True
    
    def test_invalid_mesh_type(self):
        """Test validation error for invalid mesh type."""
        with pytest.raises(ValueError, match="Expected Mesh object"):
            validate_mesh_setup("not_a_mesh")
    
    def test_mesh_no_devices(self):
        """Test validation error for mesh with no devices."""
        mock_mesh = Mock(spec=Mesh)
        mock_mesh.devices.flatten.return_value = []
        
        with pytest.raises(ValueError, match="Mesh has no devices"):
            validate_mesh_setup(mock_mesh)
    
    def test_mesh_no_axis_names(self):
        """Test validation error for mesh with no axis names."""
        mock_mesh = Mock(spec=Mesh)
        mock_mesh.devices.flatten.return_value = [Mock(platform='tpu')]
        mock_mesh.axis_names = []
        
        with pytest.raises(ValueError, match="Mesh has no axis names"):
            validate_mesh_setup(mock_mesh)
    
    def test_mesh_duplicate_axis_names(self):
        """Test validation error for duplicate axis names."""
        mock_mesh = Mock(spec=Mesh)
        mock_mesh.devices.flatten.return_value = [Mock(platform='tpu')]
        mock_mesh.axis_names = ('x', 'x', 'y')
        
        with pytest.raises(ValueError, match="Duplicate axis names"):
            validate_mesh_setup(mock_mesh)


class TestMeshInfo:
    """Test mesh information functionality."""
    
    def test_get_mesh_info(self):
        """Test getting mesh information."""
        mock_devices = [
            Mock(platform='tpu', process_index=0),
            Mock(platform='tpu', process_index=1)
        ]
        
        mock_mesh = Mock(spec=Mesh)
        mock_mesh.shape = (2, 4)
        mock_mesh.axis_names = ('x', 'y')
        mock_mesh.devices.flatten.return_value = mock_devices
        mock_mesh.devices.shape = (2, 4)
        
        info = get_mesh_info(mock_mesh)
        
        assert info['shape'] == (2, 4)
        assert info['axis_names'] == ('x', 'y')
        assert info['device_count'] == 2
        assert info['device_platforms'] == ['tpu']
        assert info['device_array_shape'] == (2, 4)
        assert info['is_multi_host'] is True
        assert info['process_indices'] == [0, 1]
        assert info['tpu_topology'] == (2, 4)
        assert info['is_tpu_pod'] is False  # <= 8 devices
    
    def test_tpu_pod_detection(self):
        """Test TPU pod detection for large device counts."""
        mock_devices = [Mock(platform='tpu', process_index=0) for _ in range(32)]
        
        mock_mesh = Mock(spec=Mesh)
        mock_mesh.shape = (4, 4, 2)
        mock_mesh.axis_names = ('x', 'y', 'z')
        mock_mesh.devices.flatten.return_value = mock_devices
        
        info = get_mesh_info(mock_mesh)
        
        assert info['is_tpu_pod'] is True  # > 8 devices


class TestSetupTPUMesh:
    """Test high-level TPU mesh setup function."""
    
    def setup_method(self):
        """Clear global mesh before each test."""
        clear_global_mesh()
    
    @patch('sharding.mesh_setup.make_or_get_global_mesh')
    @patch('sharding.mesh_setup.validate_mesh_setup')
    @patch('sharding.mesh_setup.print_mesh_info')
    def test_successful_setup(self, mock_print, mock_validate, mock_make_mesh):
        """Test successful TPU mesh setup."""
        mock_mesh = Mock()
        mock_make_mesh.return_value = mock_mesh
        mock_validate.return_value = True
        
        result = setup_tpu_mesh()
        
        mock_make_mesh.assert_called_once()
        mock_validate.assert_called_once_with(mock_mesh)
        mock_print.assert_called_once_with(mock_mesh)
        assert result == mock_mesh
    
    @patch('sharding.mesh_setup.make_mesh')
    def test_setup_failure(self, mock_make_mesh):
        """Test TPU mesh setup failure."""
        mock_make_mesh.side_effect = Exception("Setup failed")
        
        with pytest.raises(RuntimeError, match="TPU mesh setup failed"):
            setup_tpu_mesh(use_global=False)
    
    @patch('sharding.mesh_setup.make_or_get_global_mesh')
    @patch('sharding.mesh_setup.validate_mesh_setup')
    @patch('sharding.mesh_setup.print_mesh_info')
    def test_setup_options(self, mock_print, mock_validate, mock_make_mesh):
        """Test setup with different options."""
        mock_mesh = Mock()
        mock_make_mesh.return_value = mock_mesh
        
        # Test with validation and printing disabled
        setup_tpu_mesh(validate=False, print_info=False)
        
        mock_validate.assert_not_called()
        mock_print.assert_not_called()


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_print_env_usage(self, capsys):
        """Test printing environment usage information."""
        print_env_usage()
        
        captured = capsys.readouterr()
        assert "TPU Mesh Environment Variable Configuration" in captured.out
        assert "TPU_MESH_CONFIG" in captured.out
        assert "Examples:" in captured.out


class TestIntegration:
    """Integration tests for complete workflows."""
    
    def setup_method(self):
        """Clear environment and global mesh."""
        clear_global_mesh()
        env_vars = [
            'TPU_MESH_CONFIG', 'TPU_AXIS_NAMES', 'TPU_DEVICE_COUNT',
            'TPU_FORCE_TOPOLOGY', 'TPU_USE_GLOBAL_MESH'
        ]
        for var in env_vars:
            if var in os.environ:
                del os.environ[var]
    
    @patch('jax.devices')
    def test_complete_workflow_with_env_config(self, mock_devices):
        """Test complete workflow with environment configuration."""
        # Setup environment
        os.environ['TPU_MESH_CONFIG'] = '4,4,2'
        os.environ['TPU_AXIS_NAMES'] = 'x,y,z'
        os.environ['TPU_USE_GLOBAL_MESH'] = 'true'
        
        # Mock devices
        mock_devices.return_value = [Mock(platform='tpu') for _ in range(32)]
        
        with patch('sharding.mesh_setup.create_device_mesh_with_assertions') as mock_create:
            mock_mesh = Mock()
            mock_mesh.shape = (4, 4, 2)
            mock_mesh.axis_names = ('x', 'y', 'z')
            mock_mesh.devices.flatten.return_value = [Mock(platform='tpu') for _ in range(32)]
            mock_create.return_value = mock_mesh
            
            # First call should create and store mesh
            result1 = setup_tpu_mesh()
            assert result1 == mock_mesh
            assert get_global_mesh() == mock_mesh
            
            # Second call should reuse global mesh
            result2 = setup_tpu_mesh()
            assert result2 == mock_mesh
            
            # Should only create mesh once
            mock_create.assert_called_once()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])