import io
from dataclasses import dataclass
from typing import List
import torch
import torch.nn as nn
import tenseal as ts

from medshield.model.serialization import (
    model_to_critical_vector,
    model_to_non_critical_vector,
    critical_vector_to_model,
    non_critical_vector_to_model,
)
from medshield.crypto.encryption import encrypt_vector, decrypt_vector

@dataclass
class SelectiveUpdate:
    """A hybrid payload containing encrypted critical parameters and plaintext non-critical parameters."""
    encrypted_critical: List[ts.CKKSVector]
    plaintext_non_critical: torch.Tensor
    
    def serialize(self) -> bytes:
        """Serialize the update into bytes.
        
        The plaintext part uses torch.save on a BytesIO object for efficient packing.
        """
        buffer = io.BytesIO()
        
        # We need to serialize the CKKSVectors
        serialized_critical = [vec.serialize() for vec in self.encrypted_critical]
        
        # Save both together
        payload = {
            'encrypted_critical': serialized_critical,
            'plaintext_non_critical': self.plaintext_non_critical.cpu()
        }
        
        torch.save(payload, buffer)
        return buffer.getvalue()
    
    @classmethod
    def deserialize(cls, data: bytes, public_context: ts.Context) -> 'SelectiveUpdate':
        """Deserialize bytes back into a SelectiveUpdate."""
        buffer = io.BytesIO(data)
        payload = torch.load(buffer, weights_only=False)
        
        serialized_critical = payload['encrypted_critical']
        plaintext_non_critical = payload['plaintext_non_critical']
        
        encrypted_critical = [ts.ckks_vector_from(public_context, vec_bytes) for vec_bytes in serialized_critical]
        
        return cls(
            encrypted_critical=encrypted_critical,
            plaintext_non_critical=plaintext_non_critical
        )


def create_selective_update(model: nn.Module, public_context: ts.Context) -> SelectiveUpdate:
    """Extract and selectively encrypt a model's weights into a hybrid payload."""
    critical_vec = model_to_critical_vector(model)
    non_critical_vec = model_to_non_critical_vector(model)
    
    # Encrypt ONLY the critical vector
    encrypted_critical = encrypt_vector(public_context, critical_vec)
    
    # Non-critical is left as plaintext tensor (half-precision to save bandwidth if possible, 
    # but we will just keep as float32 for simplicity, torch handles compression)
    return SelectiveUpdate(
        encrypted_critical=encrypted_critical,
        plaintext_non_critical=non_critical_vec
    )


def apply_selective_update(update: SelectiveUpdate, model: nn.Module, secret_context: ts.Context) -> None:
    """Decrypt the critical parameters and apply both critical and non-critical to the model."""
    # Ensure context is linked
    for chunk in update.encrypted_critical:
        chunk.link_context(secret_context)
        
    decrypted_critical_list = decrypt_vector(update.encrypted_critical)
    decrypted_critical_tensor = torch.tensor(decrypted_critical_list, dtype=torch.float32)
    
    # Apply to model
    non_critical_vector_to_model(update.plaintext_non_critical, model)
    critical_vector_to_model(decrypted_critical_tensor, model)
