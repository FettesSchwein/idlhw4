import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        # TODO: Implement forward pass
        # Compute the softmax in a numerically stable way
        # Apply it to the dimension specified by the `dim` parameter
        Z_stable = Z - np.max(Z, axis=self.dim, keepdims=True)
        exp_Z = np.exp(Z_stable)
        sum_exp_Z = np.sum(exp_Z, axis=self.dim, keepdims=True)
        self.A = exp_Z / sum_exp_Z
        
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # TODO: Implement backward pass
        
        # Get the shape of the input
        og_shape = self.A.shape
        # Find the dimension along which softmax was applied
        dim_idx=self.dim%len(og_shape)
           
        # Reshape input to 2D
        A_moved = np.moveaxis(self.A, dim_idx, -1)
        dldA_moved = np.moveaxis(dLdA, dim_idx, -1)
        moved_shape = A_moved.shape
        C=moved_shape[-1]

        A_flat = A_moved.reshape(-1, C)
        dLdA_flat = dldA_moved.reshape(-1, C)

        dot_product = np.sum(A_flat * dLdA_flat, axis=1, keepdims=True)
        dLdZ_flat = A_flat * (dLdA_flat - dot_product)
        dLdZ_moved = dLdZ_flat.reshape(moved_shape)
        dLdZ = np.moveaxis(dLdZ_moved, -1, dim_idx)
        return dLdZ
        # if len(shape) > 2:
        #     self.A = NotImplementedError
        #     dLdA = NotImplementedError

        # # Reshape back to original dimensions if necessary
        # if len(shape) > 2:
        #     # Restore shapes to original
        #     self.A = NotImplementedError
        #     dLdZ = NotImplementedError

        # raise NotImplementedError
 

    