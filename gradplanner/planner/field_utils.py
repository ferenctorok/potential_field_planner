import numpy as np


class Pixel:
    """Class representing a grid pixel."""
    def __init__(self, 
                 x,
                 y,
                 value=0,
                 grad=np.array([0, 0]),
                 parent=None
                 ):
        """Initializes a pixel."""
        self.x = x
        self.y = y
        self.value = value
        self.grad = grad
        # if the pixel is used in a RepulsiveField object, it is worth it to know,
        # which obstacle pixel influences it. It is the index of its parent.
        self.parent = parent
    
    
    def normalize_grad(self):
        """Normalizes its gradient."""
        self.scale_grad()
        
        
    def scale_grad(self, L=1.0, eps=1e-6):
        """Scales its gradient to be the lenght specified by L"""
        length = np.linalg.norm(self.grad)
        if length > eps:
            self.grad = self.grad / length * L
        else:
            self_grad = np.array([0, 0])


def combine_grad_fields(field1, field2):
    """
    Combines two gradient fields by summing the gradiends in every point. 
    The absolute values of each pixel are not interesting.
    Inputs:
        - field1: np.array(N, M) of Pixels.
        - field2: np.array(N, M) of Pixels.
    Output:
        - out_field: np.array(N, M) of Pixels.
    """
    
    assert field1.shape[0] == field2.shape[0], "field1.shape[0] != field2.shape[0]"
    assert field1.shape[1] == field2.shape[1], "field1.shape[1] != field2.shape[1]"
    
    out_field = np.ndarray(field1.shape, dtype=np.object)
    N, M = field1.shape
    
    for i in range(N):
        for j in range(M):
            grad = field1[i, j].grad + field2[i, j].grad
            out_field[i, j] = Pixel(i, j, 0, grad)
            out_field[i, j].normalize_grad()
    
    return out_field


def get_values_from_field(field):
    """Returns an (N, M) np.array with the values of the pixels in the field.
    Input:
        - field: np.array(N, M) of Pixels.
    Output:
        - out: np.array(N, M) with the values.
    """

    N, M = field.shape
    out = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            out[i, j] = field[i, j].value
    
    return out