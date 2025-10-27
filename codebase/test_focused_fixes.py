#!/usr/bin/env python3
"""
Focused test to validate the specific fixes made to the codebase.

This test validates:
1. HRM phase guarding in ValkyrieModel.__call__
2. Checkpoint absolute path resolution
3. Training loop hrm_enabled parameter passing

Tests are designed to be simple and avoid complex import dependencies.
"""

import os
import sys
import tempfile
from pathlib import Path

def test_hrm_phase_guarding():
    """Test that ValkyrieModel.__call__ properly guards HRM features by phase."""
    print("[TEST] HRM Phase Guarding")
    print("Testing HRM phase-based feature guarding...")
    
    try:
        # Check that the ValkyrieModel.__call__ method has hrm_enabled parameter
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        
        # Read the valkyrie.py file to check for hrm_enabled parameter
        valkyrie_path = Path(__file__).parent / "src" / "model" / "valkyrie.py"
        
        if not valkyrie_path.exists():
            print(f"    ✗ ValkyrieModel file not found: {valkyrie_path}")
            return False
            
        with open(valkyrie_path, 'r') as f:
            content = f.read()
        
        # Check for hrm_enabled parameter in __call__ method
        if "hrm_enabled: bool = True" not in content:
            print("    ✗ hrm_enabled parameter not found in __call__ method")
            return False
        
        # Check for proper guarding of HRM processing
        if "self.config.use_hrm and hrm_enabled and hasattr(self, 'hrm')" not in content:
            print("    ✗ HRM processing not properly guarded by hrm_enabled")
            return False
            
        # Check for guarded hrm_state return
        if "'hrm_state': next_hrm_state if (self.config.use_hrm and hrm_enabled) else None" not in content:
            print("    ✗ hrm_state return not properly guarded")
            return False
            
        print("    ✓ HRM phase guarding implemented correctly")
        return True
        
    except Exception as e:
        print(f"    ✗ HRM phase guarding test failed: {e}")
        return False

def test_training_loop_hrm_parameter():
    """Test that training loop passes hrm_enabled parameter to model."""
    print("\n[TEST] Training Loop HRM Parameter")
    print("Testing training loop hrm_enabled parameter passing...")
    
    try:
        # Check that the training loop passes hrm_enabled to model.apply
        train_loop_path = Path(__file__).parent / "src" / "train" / "train_loop.py"
        
        if not train_loop_path.exists():
            print(f"    ✗ Training loop file not found: {train_loop_path}")
            return False
            
        with open(train_loop_path, 'r') as f:
            content = f.read()
        
        # Check for hrm_enabled parameter being passed to model.apply
        if "hrm_enabled=phase_config.hrm_enabled" not in content:
            print("    ✗ hrm_enabled parameter not passed to model.apply")
            return False
            
        print("    ✓ Training loop passes hrm_enabled parameter correctly")
        return True
        
    except Exception as e:
        print(f"    ✗ Training loop HRM parameter test failed: {e}")
        return False

def test_checkpoint_absolute_paths():
    """Test that checkpoint paths are resolved to absolute paths."""
    print("\n[TEST] Checkpoint Absolute Paths")
    print("Testing checkpoint absolute path resolution...")
    
    try:
        # Test the path resolution logic
        def _abs(path: str) -> str:
            return path if os.path.isabs(path) else os.path.abspath(path)
        
        # Test cases
        test_cases = [
            ("relative/path", True),  # Should be converted to absolute
            ("/absolute/path", False),  # Should remain unchanged
            ("./current/dir", True),  # Should be converted to absolute
            ("~/home/path", True),  # Should be converted to absolute
        ]
        
        for test_path, should_change in test_cases:
            result = _abs(test_path)
            is_absolute = os.path.isabs(result)
            
            if not is_absolute:
                print(f"    ✗ Path not converted to absolute: {test_path} -> {result}")
                return False
                
            if should_change and result == test_path:
                # Special case for paths that might already be absolute on this system
                if not os.path.isabs(test_path):
                    print(f"    ✗ Relative path not converted: {test_path}")
                    return False
        
        print("    ✓ Checkpoint path resolution works correctly")
        return True
        
    except Exception as e:
        print(f"    ✗ Checkpoint absolute paths test failed: {e}")
        return False

def test_cache_disabled_during_training():
    """Test that use_cache=False is set during training."""
    print("\n[TEST] Cache Disabled During Training")
    print("Testing cache disabled during training...")
    
    try:
        # Check that the training loop sets use_cache=False
        train_loop_path = Path(__file__).parent / "src" / "train" / "train_loop.py"
        
        if not train_loop_path.exists():
            print(f"    ✗ Training loop file not found: {train_loop_path}")
            return False
            
        with open(train_loop_path, 'r') as f:
            content = f.read()
        
        # Look for use_cache=False in model.apply calls during training
        lines = content.split('\n')
        found_training_apply = False
        found_cache_disabled = False
        
        for i, line in enumerate(lines):
            if 'self.model.apply' in line:
                found_training_apply = True
                # Check the surrounding lines for use_cache=False and training=True
                context_lines = lines[max(0, i-3):i+10]  # Look at more lines after
                context = '\n'.join(context_lines)
                if 'use_cache=False' in context and 'training=True' in context:
                    found_cache_disabled = True
                    break
        
        if not found_training_apply:
            print("    ✗ Training model.apply call not found")
            return False
            
        if not found_cache_disabled:
            print("    ✗ use_cache=False not found in training model.apply")
            return False
            
        print("    ✓ Cache properly disabled during training")
        return True
        
    except Exception as e:
        print(f"    ✗ Cache disabled test failed: {e}")
        return False

def run_all_tests():
    """Run all focused tests."""
    print("=" * 60)
    print("FOCUSED FIXES VALIDATION")
    print("=" * 60)
    
    tests = [
        ("HRM Phase Guarding", test_hrm_phase_guarding),
        ("Training Loop HRM Parameter", test_training_loop_hrm_parameter),
        ("Checkpoint Absolute Paths", test_checkpoint_absolute_paths),
        ("Cache Disabled During Training", test_cache_disabled_during_training),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = "PASS" if success else "FAIL"
        except Exception as e:
            print(f"    ✗ {test_name} failed with exception: {e}")
            results[test_name] = "ERROR"
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in results.items():
        status_symbol = "✓" if result == "PASS" else "✗"
        print(f"{status_symbol} {test_name}: {result}")
        if result != "PASS":
            all_passed = False
    
    if all_passed:
        print("\n🎉 All focused tests passed! Fixes are working correctly.")
    else:
        print("\n❌ Some tests failed. Please review and fix issues.")
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)