from typing import Union

import numpy as np
import torch


def is_rotation_matrix(matrix: np.ndarray):
    # Taken from https://stackoverflow.com/questions/53808503/how-to-test-if-a-matrix-is-a-rotation-matrix
    I = np.identity(matrix.shape[0])
    return np.allclose((np.matmul(matrix, matrix.T)), I, atol=1e-5) and (
        np.isclose(np.linalg.det(matrix), 1, atol=1e-8))

def ensure_homogenized(vector_or_matrix: Union[torch.Tensor, np.ndarray]):
    if isinstance(vector_or_matrix, torch.Tensor):
        if len(vector_or_matrix) == 1 and vector_or_matrix.shape[0] == 3:
            # (3,) Vector
            vector_or_matrix = torch.concat([vector_or_matrix, torch.ones((1,))])

        elif vector_or_matrix.shape[1] == 3:
            # N x 3 Matrix
            vector_or_matrix = torch.concat([vector_or_matrix, torch.ones((vector_or_matrix.shape[0], 1))], dim=1)
    else:
        if len(vector_or_matrix) == 1 and vector_or_matrix.shape[0] == 3:
            # (3,) Vector
            vector_or_matrix = np.concatenate([vector_or_matrix, np.ones((1,))])

        elif vector_or_matrix.shape[1] == 3:
            # N x 3 Matrix
            vector_or_matrix = np.concatenate([vector_or_matrix, np.ones((vector_or_matrix.shape[0], 1))], axis=1)

    return vector_or_matrix
