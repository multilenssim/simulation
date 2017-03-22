from chroma.geometry import Mesh
import numpy as np

def shift(mesh, shift):
    #input shift as a vector
    vertices = mesh.vertices
    triangles = mesh.triangles
    n = np.shape(vertices)[0]
    newvertices = vertices + np.tile(shift, (n, 1))
    return Mesh(newvertices, triangles, remove_duplicate_vertices=True)

def rotate(mesh, rotation_matrix):
    vertices = mesh.vertices
    triangles = mesh.triangles
    n = np.shape(vertices)[0]
    newvertices = np.empty((n, 3))
    for i in range(n):
        newvertices[i] = np.dot(rotation_matrix, vertices[i])
    return Mesh(newvertices, triangles, remove_duplicate_vertices=True)

