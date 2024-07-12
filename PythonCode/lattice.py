import numpy as np

class Lattice:
    def __init__(self, dim, vectors, rvectors, rv_norm, vol, r_vol, b_mat_inv):
        self.dim = dim
        self.vectors = vectors
        self.rvectors = rvectors
        self.rv_norm = rv_norm
        self.vol = vol
        self.r_vol = r_vol
        self.b_mat_inv = b_mat_inv

def set_Lattice(dim, vectors):
    # Making primitive vectors
    pvector_1 = np.zeros(3)
    pvector_2 = np.zeros(3)
    pvector_3 = np.zeros(3)

    if dim == 1:
        pvector_1[0] = vectors[0][0]
        pvector_2[1] = 1.0  # 0 1 0
        pvector_3[2] = 1.0  # 0 0 1
    elif dim == 2:
        pvector_1[0:2] = vectors[0][0:2]
        pvector_2[0:2] = vectors[1][0:2]
        pvector_3[2] = 1.0  # 0 0 1
    elif dim == 3:
        pvector_1[0:3] = vectors[0][0:3]
        pvector_2[0:3] = vectors[1][0:3]
        pvector_3[0:3] = vectors[2][0:3]

    # Making reciprocal lattice vectors
    vol = np.dot(pvector_1, np.cross(pvector_2, pvector_3))
    rvector_1 = 2 * np.pi * np.cross(pvector_2, pvector_3) / vol
    rvector_2 = 2 * np.pi * np.cross(pvector_3, pvector_1) / vol
    rvector_3 = 2 * np.pi * np.cross(pvector_1, pvector_2) / vol

    if vol < 0.0:
        print("Axis vectors are left handed")
        vol = abs(vol)

    rv_norm = np.zeros(dim)

    if dim == 1:
        rvectors = [rvector_1[0]]
        rv_norm[0] = np.linalg.norm(rvector_1[0])
    elif dim == 2:
        rvectors = [rvector_1[0:2], rvector_2[0:2]]
        rv_norm[0] = np.linalg.norm(rvector_1[0:2])
        rv_norm[1] = np.linalg.norm(rvector_1[0:2])
    elif dim == 3:
        rvectors = [rvector_1[0:3], rvector_2[0:3], rvector_3[0:3]]
        rv_norm[0] = np.linalg.norm(rvector_1[0:3])
        rv_norm[1] = np.linalg.norm(rvector_1[0:3])
        rv_norm[2] = np.linalg.norm(rvector_1[0:3])

    r_vol = (2 * np.pi)**3 / vol

    print("Lattice vectors : ")
    for i in range(dim):
        print(vectors[i][0:dim])
    print("Reciprocal lattice vectors : ")
    for i in range(dim):
        print(rvectors[i][0:dim])

    b_mat_inv = np.zeros((dim, dim))
    for i in range(dim):
        b_mat_inv[:, i] = rvectors[i][0:dim]
    b_mat_inv = np.linalg.inv(b_mat_inv)

    print("Direct lattice volume     : ", vol, " [a.u.]")
    print("Reciprocal lattice volume : ", r_vol, " [a.u.]")

    lattice = Lattice(
        dim,
        vectors,
        rvectors,
        rv_norm,
        vol,
        r_vol,
        b_mat_inv
    )
    return lattice
