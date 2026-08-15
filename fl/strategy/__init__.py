"""Custom Flower Strategy performing FedAvg on CKKS ciphertexts."""
from .encrypted_fedavg import EncryptedFedAvg

__all__ = ["EncryptedFedAvg"]
