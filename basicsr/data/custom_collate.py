import torch
import torch.utils.data as data
from torch.utils.data._utils.collate import default_collate


def safe_collate_fn(batch):
    """
    Custom collate function that ensures all tensors have contiguous memory layout
    and handles potential tensor creation issues.
    """
    if not batch:
        return batch
    
    # Get the first item to understand the structure
    first_item = batch[0]
    
    if isinstance(first_item, dict):
        # Handle dictionary batches
        result = {}
        for key in first_item.keys():
            values = [item[key] for item in batch]
            
            # Check if all values are tensors
            if all(isinstance(v, torch.Tensor) for v in values):
                # Ensure all tensors are contiguous before collating
                contiguous_values = []
                for v in values:
                    if not v.is_contiguous():
                        v = v.contiguous()
                    contiguous_values.append(v)
                
                try:
                    result[key] = default_collate(contiguous_values)
                except RuntimeError as e:
                    if "Trying to resize storage that is not resizable" in str(e):
                        # Handle the specific error by creating new tensors
                        print(f"Warning: Storage resize error for key '{key}', creating new tensors")
                        # Stack tensors manually to avoid storage issues
                        stacked = torch.stack([v.clone() for v in contiguous_values])
                        result[key] = stacked
                    elif "stack expects each tensor to be equal size" in str(e):
                        # Handle tensor size mismatch - pad tensors to same size
                        print(f"Warning: Tensor size mismatch for key '{key}', padding to same size")
                        # Find the maximum dimensions
                        max_dims = [max(t.shape[i] for t in contiguous_values) for i in range(len(contiguous_values[0].shape))]
                        
                        # Pad all tensors to the same size
                        padded_values = []
                        for tensor in contiguous_values:
                            # Calculate padding needed
                            pad_dims = []
                            for i in range(len(tensor.shape)):
                                pad_needed = max_dims[i] - tensor.shape[i]
                                pad_dims.extend([0, pad_needed])
                            
                            # Pad the tensor
                            padded_tensor = torch.nn.functional.pad(tensor, pad_dims, mode='reflect')
                            padded_values.append(padded_tensor)
                        
                        # Now try to stack the padded tensors
                        try:
                            result[key] = torch.stack(padded_values)
                        except:
                            # If still fails, return as list
                            result[key] = padded_values
                    else:
                        raise e
            else:
                # Use default collate for non-tensor values
                result[key] = default_collate(values)
        
        return result
    else:
        # Handle non-dictionary batches
        return default_collate(batch)


def create_safe_dataloader(dataset, **kwargs):
    """
    Create a DataLoader with the safe collate function.
    """
    # Set the custom collate function
    kwargs['collate_fn'] = safe_collate_fn
    
    # Ensure num_workers is reasonable to avoid multiprocessing issues
    if 'num_workers' not in kwargs:
        kwargs['num_workers'] = 0  # Use single process to avoid multiprocessing issues
    
    return data.DataLoader(dataset, **kwargs)
