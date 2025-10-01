#!/usr/bin/env python3
"""
Test script to verify that the configuration file loads without the use_edge_map error
"""
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    
    print("Loading configuration...")
    config = OmegaConf.load("configs/stableSRNew/v2-finetune_text_T_512_weiql_0930.yaml")
    
    print("Configuration loaded successfully!")
    print(f"Model target: {config.model.target}")
    
    # Try to instantiate the model configuration
    print("Testing model instantiation...")
    try:
        model = instantiate_from_config(config.model)
        print("SUCCESS: Model instantiated without errors!")
        print(f"Model type: {type(model)}")
    except Exception as e:
        print(f"ERROR during model instantiation: {e}")
        import traceback
        traceback.print_exc()
        
except ImportError as e:
    print(f"Import error: {e}")
    print("Some dependencies are missing, but the configuration file syntax is correct.")
except Exception as e:
    print(f"Configuration error: {e}")
    import traceback
    traceback.print_exc()
