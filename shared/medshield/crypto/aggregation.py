from typing import List, cast
import torch
import tenseal as ts
from medshield.crypto.selective import SelectiveUpdate

def add_encrypted_vectors(list_of_vectors: List[List[ts.CKKSVector]]) -> List[ts.CKKSVector]:
    """
    Homomorphically adds corresponding chunks of multiple encrypted weight vectors.
    
    Args:
        list_of_vectors: A list where each element is an encrypted vector 
                         (which is itself a list of CKKSVector chunks).
                         
    Returns:
        A single list of CKKSVector chunks representing the sum.
    """
    if not list_of_vectors:
        raise ValueError("Cannot aggregate an empty list of vectors.")
        
    num_updates = len(list_of_vectors)
    num_chunks = len(list_of_vectors[0])
    
    # Verify all vectors have the same number of chunks
    for vec in list_of_vectors:
        if len(vec) != num_chunks:
            raise ValueError(f"Chunk size mismatch: expected {num_chunks} chunks, got {len(vec)}.")

    # Initialize the sum with the first vector's chunks
    # Note: we use copy() to avoid modifying the original first vector in-place
    summed_chunks = [chunk.copy() for chunk in list_of_vectors[0]]
    
    # Homomorphically add the remaining vectors
    for update_idx in range(1, num_updates):
        for chunk_idx in range(num_chunks):
            # TenSEAL overloads the + operator to perform homomorphic addition
            summed_chunks[chunk_idx] += list_of_vectors[update_idx][chunk_idx]
            
    return summed_chunks

def add_selective_updates(updates: List[SelectiveUpdate]) -> SelectiveUpdate:
    """
    Aggregates multiple SelectiveUpdate objects by summing their encrypted 
    critical parts homomorphically and their plaintext non-critical parts normally.
    
    Args:
        updates: A list of SelectiveUpdate objects.
        
    Returns:
        A new SelectiveUpdate representing the sum.
    """
    if not updates:
        raise ValueError("Cannot aggregate an empty list of SelectiveUpdates.")
        
    # Extract all encrypted parts and sum them
    encrypted_parts = [u.encrypted_critical for u in updates]
    summed_encrypted_critical = add_encrypted_vectors(encrypted_parts)
    
    # Extract all plaintext parts and sum them
    # PyTorch allows elegant stacking and summing over dimension 0
    plaintext_parts = [u.plaintext_non_critical for u in updates]
    summed_plaintext_non_critical = torch.stack(plaintext_parts).sum(dim=0)
    
    return SelectiveUpdate(
        encrypted_critical=summed_encrypted_critical,
        plaintext_non_critical=summed_plaintext_non_critical
    )

def multiply_encrypted_vector_by_scalar(vector: List[ts.CKKSVector], scalar: float) -> List[ts.CKKSVector]:
    """
    Homomorphically multiplies an encrypted chunked vector by a plaintext scalar.
    
    Args:
        vector: A list of CKKSVector chunks.
        scalar: The plaintext float multiplier (e.g. client weighting factor).
        
    Returns:
        A new list of CKKSVector chunks representing the scaled vector.
    """
    if not vector:
        raise ValueError("Cannot multiply an empty list of vectors.")
        
    scaled_chunks = []
    for chunk in vector:
        # TenSEAL overloads the * operator for homomorphic scalar multiplication
        scaled_chunks.append(cast(ts.CKKSVector, chunk * scalar))
        
    return scaled_chunks

def multiply_selective_update_by_scalar(update: SelectiveUpdate, scalar: float) -> SelectiveUpdate:
    """
    Multiplies a SelectiveUpdate by a plaintext scalar, scaling both the 
    encrypted critical parts and the plaintext non-critical parts.
    
    Args:
        update: The SelectiveUpdate payload.
        scalar: The plaintext float multiplier.
        
    Returns:
        A new scaled SelectiveUpdate.
    """
    scaled_encrypted = multiply_encrypted_vector_by_scalar(update.encrypted_critical, scalar)
    scaled_plaintext = update.plaintext_non_critical * scalar
    
    return SelectiveUpdate(
        encrypted_critical=scaled_encrypted,
        plaintext_non_critical=scaled_plaintext
    )
