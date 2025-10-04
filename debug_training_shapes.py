#!/usr/bin/env python3
"""
Debug script to add to your training script to monitor tensor shapes
Add this to your training script to debug tensor shapes during training
"""

def debug_tensor_shapes(self, batch, result, edge_map=None):
    """
    Debug function to add to your training_step method
    Call this right after getting the input to monitor tensor shapes
    """
    print("\n" + "="*60)
    print("DEBUG: Tensor Shapes in Training")
    print("="*60)
    
    if edge_map is not None:
        z, c, z_gt, x, gt, xrec, edge_map = result
        print(f"With edge processing:")
        print(f"  z (noisy latent): {z.shape}")
        print(f"  c (text conditioning): {c.shape}")
        print(f"  z_gt (ground truth latent): {z_gt.shape}")
        print(f"  x (low-quality image): {x.shape}")
        print(f"  gt (ground truth image): {gt.shape}")
        print(f"  xrec (reconstructed): {xrec.shape}")
        print(f"  edge_map: {edge_map.shape}")
    else:
        z, c, z_gt, x, gt, xrec = result
        print(f"Without edge processing:")
        print(f"  z (noisy latent): {z.shape}")
        print(f"  c (text conditioning): {c.shape}")
        print(f"  z_gt (ground truth latent): {z_gt.shape}")
        print(f"  x (low-quality image): {x.shape}")
        print(f"  gt (ground truth image): {gt.shape}")
        print(f"  xrec (reconstructed): {xrec.shape}")
    
    print("\nChecking struct_cond creation:")
    if hasattr(self, 'structcond_stage_model'):
        # Test struct_cond creation
        t_ori = torch.randint(0, 1000, (z.shape[0],), device=z.device).long()
        
        if hasattr(self, 'test_gt') and self.test_gt:
            struct_cond = self.structcond_stage_model(gt, t_ori)
            print(f"  struct_cond from gt: {struct_cond.shape} (should be [B, 256, H, W])")
        else:
            struct_cond = self.structcond_stage_model(x, t_ori)
            print(f"  struct_cond from x: {struct_cond.shape} (should be [B, 256, H, W])")
        
        # Verify channel count
        if struct_cond.shape[1] == 256:
            print("  ✓ struct_cond has correct 256 channels")
        else:
            print(f"  ✗ struct_cond has {struct_cond.shape[1]} channels, expected 256")
    else:
        print("  ✗ structcond_stage_model not found!")
    
    print("\nChecking SPADE expectations:")
    print("  SPADE expects input with 256 channels")
    print("  SPADE mlp_shared: Conv2d(256, 128, kernel_size=(3, 3))")
    
    print("="*60)

# Example of how to add this to your training script:
"""
# In your training_step method, add this after getting the input:

def training_step(self, batch, batch_idx, optimizer_idx=0):
    # ... existing code ...
    
    # Get input with edge map
    if self.use_edge_processing and 'img_edge' in batch:
        result = self.get_input(batch, self.first_stage_key, return_first_stage_outputs=True)
        z, c, z_gt, x, gt, xrec, edge_map = result
    else:
        result = self.get_input(batch, self.first_stage_key, return_first_stage_outputs=True)
        z, c, z_gt, x, gt, xrec = result
        edge_map = None
    
    # Add this debug call:
    if batch_idx % 100 == 0:  # Only debug every 100 steps to avoid spam
        debug_tensor_shapes(self, batch, result, edge_map)
    
    # ... rest of your training code ...
"""
