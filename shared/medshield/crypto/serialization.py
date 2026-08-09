import logging
import tenseal as ts
from typing import List

logger = logging.getLogger(__name__)

def serialize_encrypted_vector(vector: List[ts.CKKSVector]) -> List[bytes]:
    """Serialize a list of CKKSVector chunks into a list of bytes.
    
    Logs the total serialized size to satisfy Task 43.
    """
    serialized = []
    total_size = 0
    for chunk in vector:
        b = chunk.serialize()
        serialized.append(b)
        total_size += len(b)
        
    logger.info(f"Serialized encrypted vector size: {total_size / 1024 / 1024:.2f} MB")
    return serialized

def deserialize_encrypted_vector(
    public_context: ts.Context, serialized_vector: List[bytes]
) -> List[ts.CKKSVector]:
    """Deserialize a list of bytes into a list of CKKSVector chunks."""
    vector = []
    for b in serialized_vector:
        vector.append(ts.ckks_vector_from(public_context, b))
    return vector
