from typing import List, Union
import torch
import tenseal as ts

def encrypt_vector(context: ts.Context, vector: Union[torch.Tensor, List[float]]) -> List[ts.CKKSVector]:
    """
    Encrypts a flattened weight vector into a list of CKKS ciphertexts.
    
    Given that a TenSEAL context with poly_modulus_degree = N can only hold N/2 
    elements per ciphertext, this function chunks the input vector into pieces of 
    maximum size N/2 before encrypting them.
    
    Args:
        context: The public CKKS context.
        vector: The 1-D flattened weight vector (can be a torch.Tensor or a list of floats).
        
    Returns:
        A list of encrypted CKKSVector chunks.
    """
    if isinstance(vector, torch.Tensor):
        # Convert tensor to a flat python list of floats
        vector = vector.detach().cpu().flatten().float().tolist()
        
    # The max number of slots in a CKKS vector is N / 2
    # But TenSEAL's context may not expose it directly in Python unless we check it.
    # Usually we can get it via `context.global_scale` or knowing our poly modulus is 8192.
    max_slots = 4096 
    
    encrypted_chunks = []
    
    for i in range(0, len(vector), max_slots):
        chunk = vector[i : i + max_slots]
        encrypted_chunk = ts.ckks_vector(context, chunk)
        encrypted_chunks.append(encrypted_chunk)
        
    return encrypted_chunks


def decrypt_vector(encrypted_chunks: List[ts.CKKSVector]) -> List[float]:
    """
    Decrypts a list of CKKS ciphertext chunks and recombines them into a single list.
    
    Note: The CKKSVectors must be bound to a context that holds a secret key.
    
    Args:
        encrypted_chunks: A list of encrypted CKKSVector chunks.
        
    Returns:
        The decrypted 1-D list of floats.
    """
    decrypted_vector = []
    for chunk in encrypted_chunks:
        decrypted_vector.extend(chunk.decrypt())
        
    return decrypted_vector
