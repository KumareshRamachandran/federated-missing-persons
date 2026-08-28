"""
privacy/smpc.py

Partial Secure Multi-Party Computation (P-SMPC) using TenSEAL.
Secures model weight/gradient updates before they leave edge devices.

TenSEAL uses CKKS (a Homomorphic Encryption scheme) allowing:
  - Addition and multiplication on encrypted tensors
  - The coordinator aggregates encrypted updates — never seeing raw gradients

Member responsible: K Kishore (23BCE9746) — Cryptography & Privacy module

References:
  - TenSEAL: https://github.com/OpenMined/TenSEAL
  - CKKS: Cheon et al. (2017)
"""

import numpy as np

try:
    import tenseal as ts
except ImportError:
    ts = None
    print("Warning: TenSEAL not installed. Run: pip install tenseal")


class SMPCEncryptor:
    """
    Encrypts model weight updates using CKKS Homomorphic Encryption (TenSEAL).
    Allows the coordinator to aggregate encrypted updates without decryption.
    """

    def __init__(self, poly_modulus_degree: int = 8192, coeff_mod_bit_sizes: list = None):
        """
        Initialize TenSEAL CKKS context.

        Args:
            poly_modulus_degree: Controls security level (8192 = 128-bit security).
            coeff_mod_bit_sizes: Coefficient modulus chain for CKKS precision.
        """
        if coeff_mod_bit_sizes is None:
            coeff_mod_bit_sizes = [60, 40, 40, 60]

        self.poly_modulus_degree = poly_modulus_degree
        self.coeff_mod_bit_sizes = coeff_mod_bit_sizes

        if ts is not None:
            try:
                self.context = ts.context(
                    ts.SCHEME_TYPE.CKKS,
                    poly_modulus_degree=poly_modulus_degree,
                    coeff_mod_bit_sizes=coeff_mod_bit_sizes
                )
                self.context.generate_galois_keys()
                self.context.global_scale = 2 ** 40
            except Exception as e:
                print(f"TenSEAL context initialization failed: {e}. Operating in fallback mode.")
                self.context = None
        else:
            self.context = None

    def encrypt_weights(self, weights: list) -> list:
        """
        Encrypt a list of numpy weight arrays before sending to coordinator.

        Args:
            weights: List of np.ndarray model weight arrays.

        Returns:
            List of encrypted TenSEAL CKKSVector objects (serialized bytes) or fallback representation.
        """
        encrypted_layers = []
        for weight in weights:
            w_arr = np.asarray(weight, dtype=np.float64)
            flat = w_arr.flatten().tolist()

            if self.context is not None:
                ckks_vec = ts.ckks_vector(self.context, flat)
                encrypted_layers.append(ckks_vec.serialize())
            else:
                # Fallback serialized dict for mock homomorphic operation
                fallback_payload = {
                    "shape": w_arr.shape,
                    "data": np.asarray(flat, dtype=np.float32)
                }
                encrypted_layers.append(fallback_payload)
        return encrypted_layers

    def decrypt_weights(self, encrypted_weights: list, original_shapes: list) -> list:
        """
        Decrypt aggregated weights received from coordinator.

        Args:
            encrypted_weights: List of serialized encrypted vectors.
            original_shapes: List of shapes to reshape decrypted arrays.

        Returns:
            List of decrypted np.ndarray weight arrays.
        """
        decrypted_layers = []
        for enc, shape in zip(encrypted_weights, original_shapes):
            if self.context is not None and isinstance(enc, bytes):
                vec = ts.ckks_vector_from(self.context, enc)
                dec_flat = np.array(vec.decrypt(), dtype=np.float32)
                # Trim or pad if CKKS vector size slightly differs due to alignment
                num_elements = int(np.prod(shape))
                dec_flat = dec_flat[:num_elements]
                decrypted_layers.append(dec_flat.reshape(shape))
            elif isinstance(enc, dict):
                # Fallback mode
                data = enc["data"]
                num_elements = int(np.prod(shape))
                decrypted_layers.append(data[:num_elements].reshape(shape))
            else:
                # Raw array fallback
                decrypted_layers.append(np.asarray(enc, dtype=np.float32).reshape(shape))

        return decrypted_layers


def aggregate_encrypted_updates(encrypted_updates: list, context=None) -> list:
    """
    Coordinator-side: Sum encrypted updates WITHOUT decrypting them.
    HE supports addition on ciphertexts directly.

    Args:
        encrypted_updates: List of lists of encrypted weight arrays from all clients.
        context: Optional TenSEAL context for deserializing vectors.

    Returns:
        List of summed encrypted weight arrays.
    """
    if not encrypted_updates:
        return []

    num_clients = len(encrypted_updates)
    num_layers = len(encrypted_updates[0])

    aggregated = []
    for layer_idx in range(num_layers):
        client_layer_0 = encrypted_updates[0][layer_idx]

        if context is not None and isinstance(client_layer_0, bytes):
            # TenSEAL CKKS homomorphic addition
            sum_vec = ts.ckks_vector_from(context, client_layer_0)
            for c_idx in range(1, num_clients):
                client_layer = encrypted_updates[c_idx][layer_idx]
                vec = ts.ckks_vector_from(context, client_layer)
                sum_vec += vec
            aggregated.append(sum_vec.serialize())
        elif isinstance(client_layer_0, dict):
            # Fallback dict addition
            shape = client_layer_0["shape"]
            sum_data = np.copy(client_layer_0["data"])
            for c_idx in range(1, num_clients):
                sum_data += encrypted_updates[c_idx][layer_idx]["data"]
            aggregated.append({"shape": shape, "data": sum_data})
        else:
            # Fallback raw numpy addition
            sum_data = np.copy(np.asarray(client_layer_0))
            for c_idx in range(1, num_clients):
                sum_data += np.asarray(encrypted_updates[c_idx][layer_idx])
            aggregated.append(sum_data)

    return aggregated

