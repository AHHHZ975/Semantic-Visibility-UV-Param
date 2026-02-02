################################ Usage #####################################
# python repair_mesh.py ..\data\Duckie.obj ..\data\Duckie_repaired.obj
############################################################################

# This file is used to repaire and clean up the mesh objects to be fed into the OptCuts method.
# What this script exactly does:
# - Uses libigl to clean a mesh for OptCuts.
# - Preserves faces (no face deletion): duplicates vertices where needed to split non-manifold connectivity into manifold connectivity.

import sys
import numpy as np
import igl

def safe_read(path):
    # igl.read_triangle_mesh returns V,F
    V, F = igl.read_triangle_mesh(path)
    V = np.array(V, dtype=float)
    F = np.array(F, dtype=int)
    return V, F

def safe_write(path, V, F):
    # write mesh; igl.write_triangle_mesh exists in bindings
    igl.write_triangle_mesh(path, V, F)
    print("Wrote:", path)



# -robust reducer: flatten any nested container and compute "all true" safely
def flatten_all_true(x):
    """
    Recursively reduce x to a single boolean representing "all elements truthy".
    Handles:
      - scalars (bool/int/float)
      - numpy arrays
      - python lists/tuples
      - arbitrary iterables (fall back)
      - None -> False
    If an element cannot be interpreted, it is treated as False conservatively.
    """
    # None
    if x is None:
        return False

    # Numpy scalar or python scalar
    if isinstance(x, (bool, np.bool_, int, np.integer, float, np.floating, np.number)):
        try:
            return bool(x)
        except Exception:
            return False

    # Numpy array: require all elements True
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return False
        try:
            return bool(x.all())
        except Exception:
            # if it's an array of objects, iterate
            try:
                return all(flatten_all_true(e) for e in x)
            except Exception:
                return False

    # Python containers: list/tuple/set
    if isinstance(x, (list, tuple, set)):
        if len(x) == 0:
            return False
        return all(flatten_all_true(e) for e in x)

    # Generic iterable objects (e.g., libigl return types)
    try:
        it = iter(x)
    except TypeError:
        # Not iterable: try bool conversion as last resort
        try:
            return bool(x)
        except Exception:
            return False
    else:
        # iterate and reduce; be defensive for heterogeneous elements
        try:
            # convert to list to avoid exhaustion of single-pass iterators
            lst = list(it)
            if len(lst) == 0:
                return False
            return all(flatten_all_true(e) for e in lst)
        except Exception:
            return False

# prepare(...) with robust manifold tests
def prepare(V, F, dup_eps=1e-8, verbose=True):
    # 1) remove duplicate vertices (small eps)
    if verbose:
        print("Removing duplicate vertices (eps=%g)..." % dup_eps)
    try:
        SV, SVI, SVJ, SF = igl.remove_duplicate_vertices(V, F, dup_eps)
        V = np.array(SV, dtype=float)
        F = np.array(SF, dtype=int)
    except Exception:
        try:
            SV, SVI, SVJ = igl.remove_duplicate_vertices(V, dup_eps)
            # remap faces using SVJ mapping if needed
            F = np.array([[SVJ[int(i)] for i in face] for face in F], dtype=int)
            V = np.array(SV, dtype=float)
        except Exception as e:
            if verbose:
                print("Warning: remove_duplicate_vertices failed:", e)

    # 2) remove unreferenced vertices
    if verbose:
        print("Removing unreferenced vertices...")
    try:
        NV, NF, IM, J = igl.remove_unreferenced(V, F)
        V = np.array(NV, dtype=float)
        F = np.array(NF, dtype=int)
    except Exception:
        try:
            NV, NF = igl.remove_unreferenced(V, F)
            V = np.array(NV, dtype=float)
            F = np.array(NF, dtype=int)
        except Exception as e:
            if verbose:
                print("Warning: remove_unreferenced failed:", e)

    # 3) quick manifold tests (use flatten_all_true to normalize return types)
    # Try both possible signatures: igl.is_edge_manifold(F) and igl.is_edge_manifold(V,F)
    em_raw = None
    vm_raw = None
    try:
        em_raw = igl.is_edge_manifold(F)
        vm_raw = igl.is_vertex_manifold(F)
    except Exception:
        try:
            em_raw = igl.is_edge_manifold(V, F)
            vm_raw = igl.is_vertex_manifold(V, F)
        except Exception as e:
            if verbose:
                print("Warning: is_edge_manifold / is_vertex_manifold failed with both signatures:", e)
            em_raw = False
            vm_raw = False

    em = flatten_all_true(em_raw)
    vm = flatten_all_true(vm_raw)

    if verbose:
        print("Edge-manifold:", em, "Vertex-manifold:", vm)

    # 4) if not manifold, split non-manifold without deleting faces
    if not (em and vm):
        if verbose:
            print("Splitting non-manifold connectivity (duplicates vertices) ...")
        try:
            SV2, SF2, SVI2 = igl.split_nonmanifold(V, F)
            V = np.array(SV2, dtype=float)
            F = np.array(SF2, dtype=int)
        except Exception:
            try:
                SF2, SVI2 = igl.split_nonmanifold(F)
                F = np.array(SF2, dtype=int)
            except Exception as e:
                raise RuntimeError("split_nonmanifold failed: " + str(e))

        # after splitting, remove any unreferenced vertices again
        try:
            NV, NF, IM, J = igl.remove_unreferenced(V, F)
            V = np.array(NV, dtype=float)
            F = np.array(NF, dtype=int)
        except Exception:
            pass

    # 5) final manifold check
    em_raw2 = None
    vm_raw2 = None
    try:
        em_raw2 = igl.is_edge_manifold(F)
        vm_raw2 = igl.is_vertex_manifold(F)
    except Exception:
        try:
            em_raw2 = igl.is_edge_manifold(V, F)
            vm_raw2 = igl.is_vertex_manifold(V, F)
        except Exception:
            em_raw2 = False
            vm_raw2 = False

    em2 = flatten_all_true(em_raw2)
    vm2 = flatten_all_true(vm_raw2)
    if verbose:
        print("Final: Edge-manifold:", em2, "Vertex-manifold:", vm2)

    return V, F


def main(in_path, out_path):
    V, F = safe_read(in_path)
    print("Input: V =", len(V), "F =", len(F))
    Vp, Fp = prepare(V, F)
    print("Prepared: V =", len(Vp), "F =", len(Fp))
    safe_write(out_path, Vp, Fp)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python prepare_for_optcuts.py in.obj out.obj")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
