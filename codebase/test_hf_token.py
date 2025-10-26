#!/usr/bin/env python3
"""Test script to validate HuggingFace token and dataset access."""

import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, login
from datasets import load_dataset

# Load environment variables
load_dotenv()

def test_hf_token():
    """Test HuggingFace token validity and dataset access."""
    token = os.environ.get('HF_TOKEN')
    
    if not token:
        print("❌ HF_TOKEN not found in environment variables")
        return False
    
    print(f"✅ HF_TOKEN found (length: {len(token)})")
    print(f"✅ Token starts with: {token[:10]}...")
    
    # Test 1: Basic token validation
    try:
        api = HfApi(token=token)
        user_info = api.whoami()
        print(f"✅ Token is valid! Authenticated as: {user_info['name']}")
    except Exception as e:
        print(f"❌ Token validation failed: {e}")
        return False
    
    # Test 2: Try to login programmatically
    try:
        login(token=token)
        print("✅ Programmatic login successful")
    except Exception as e:
        print(f"❌ Programmatic login failed: {e}")
        return False
    
    # Test 3: Try to access specific datasets
    datasets_to_test = [
        "HuggingFaceFW/fineweb",
        "common-pile/arxiv_papers_filtered", 
        "bigcode/the-stack-dedup",
        "bigcode/starcoderdata"
    ]
    
    for dataset_name in datasets_to_test:
        try:
            print(f"🔍 Testing access to {dataset_name}...")
            # Just try to get dataset info, don't download
            dataset_info = api.dataset_info(dataset_name)
            print(f"✅ Can access {dataset_name}")
        except Exception as e:
            print(f"❌ Cannot access {dataset_name}: {e}")
    
    return True

if __name__ == "__main__":
    print("🧪 Testing HuggingFace token and dataset access...")
    print("=" * 60)
    test_hf_token()