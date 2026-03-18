import warnings
import numpy as np
from typing import Callable, Any, Optional, Tuple, Dict, List

####################################################################################################
# --------------------------------- small utils ---------------------------------
try:
    from tools import mic_vec, wrap_pos, normalize_vec
except Exception:
    from source.tools import mic_vec, wrap_pos, normalize_vec

####################################################################################################

def create_h2_candidates_by_midpoint_flip_vectorized(
    atom_coords: np.ndarray,
    H_to_OO_pairs: np.ndarray,
    box_lengths: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized construction of per-bond H2 candidates by midpoint mirroring under PBC.

    Assumes atom_coords ordering:
      - first n_O rows are oxygen positions (n_O inferred from H_to_OO_pairs)
      - next n_H rows are hydrogen positions aligned with bond index b

    Args:
        atom_coords: All-atom coordinates, shape (n_atoms, 3).
        H_to_OO_pairs: Array (n_H, 3) rows [H_index, O1, O2].
        box_lengths: Orthorhombic box lengths (3,).

    Returns:
        atomcoords_O: Wrapped oxygen positions, shape (n_O, 3).
        H2_candidates: Candidate H positions per bond, shape (n_H, 2, 3),
            ordered so that candidates[:,0] is O1-like, candidates[:,1] is O2-like.
    """
    coords = np.asarray(atom_coords, dtype=np.float64)
    pairs = np.asarray(H_to_OO_pairs, dtype=np.int64)
    L = np.asarray(box_lengths, dtype=np.float64).reshape(3)

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("atom_coords must have shape (n_atoms, 3).")
    if pairs.ndim != 2 or pairs.shape[1] != 3:
        raise ValueError("H_to_OO_pairs must have shape (n_H, 3).")
    if L.shape != (3,):
        raise ValueError("box_lengths must have shape (3,).")

    n_H = int(pairs.shape[0])
    num_O = int(pairs[:, 1:].max()) + 1

    if coords.shape[0] < num_O + n_H:
        raise ValueError("atom_coords size inconsistent with inferred num_O and n_H.")

    atomcoords_O = wrap_pos(coords[:num_O], L)
    H_pos = coords[num_O : num_O + n_H].copy()

    O1 = pairs[:, 1]
    O2 = pairs[:, 2]

    # MIC midpoint for all bonds
    r12 = mic_vec(atomcoords_O[O2] - atomcoords_O[O1], L)              # (n_H, 3)
    midpoint = atomcoords_O[O1] + 0.5 * r12                            # (n_H, 3)

    # Mirror H about midpoint under MIC
    r = mic_vec(H_pos - midpoint, L)                                   # (n_H, 3)
    H_mirror = midpoint - r                                             # (n_H, 3)

    H0 = wrap_pos(H_pos, L)                                             # (n_H, 3)
    H1 = wrap_pos(H_mirror, L)                                          # (n_H, 3)

    # Distances (MIC norms)
    def mic_norm(x: np.ndarray) -> np.ndarray:
        """MIC Euclidean norms for vectors x of shape (n,3)."""
        return np.linalg.norm(mic_vec(x, L), axis=1)

    d0_O1 = mic_norm(H0 - atomcoords_O[O1])
    d0_O2 = mic_norm(H0 - atomcoords_O[O2])
    d1_O1 = mic_norm(H1 - atomcoords_O[O1])
    d1_O2 = mic_norm(H1 - atomcoords_O[O2])

    # Masks matching your branching logic
    case_a = (d0_O1 <= d0_O2) & (d1_O2 <= d1_O1)                        # (H0->O1, H1->O2)
    case_b = (d1_O1 <= d1_O2) & (d0_O2 <= d0_O1)                        # (H1->O1, H0->O2)
    ambig = ~(case_a | case_b)

    # Allocate output and fill
    H2_candidates = np.empty((n_H, 2, 3), dtype=np.float64)

    # Case A
    H2_candidates[case_a, 0] = H0[case_a]
    H2_candidates[case_a, 1] = H1[case_a]

    # Case B
    H2_candidates[case_b, 0] = H1[case_b]
    H2_candidates[case_b, 1] = H0[case_b]

    # Ambiguous fallback: choose by smaller O1 distance
    # if d0_O1 <= d1_O1: idx0=H0 else idx0=H1
    ambig_choose_h0_as_o1 = ambig & (d0_O1 <= d1_O1)
    ambig_choose_h1_as_o1 = ambig & ~(d0_O1 <= d1_O1)

    H2_candidates[ambig_choose_h0_as_o1, 0] = H0[ambig_choose_h0_as_o1]
    H2_candidates[ambig_choose_h0_as_o1, 1] = H1[ambig_choose_h0_as_o1]

    H2_candidates[ambig_choose_h1_as_o1, 0] = H1[ambig_choose_h1_as_o1]
    H2_candidates[ambig_choose_h1_as_o1, 1] = H0[ambig_choose_h1_as_o1]

    return atomcoords_O, H2_candidates


if __name__ == "__main__":
    # Minimal deterministic demo
    box_lengths_ex = np.array([10.0, 10.0, 10.0], dtype=float)
    atomcoords_O_ex = np.array([[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]], dtype=float)
    atom_coords_ex = np.array([[1.0, 1.0, 1.0], [3.0, 1.0, 1.0], [1.8, 1.0, 1.0]], dtype=float)
    H_to_OO_pairs_ex = np.array([[0, 0, 1]], dtype=int)

    atomcoords_O_out, H2 = create_h2_candidates_by_midpoint_flip_vectorized(
        atom_coords=atom_coords_ex,
        H_to_OO_pairs=H_to_OO_pairs_ex,
        box_lengths=box_lengths_ex,
    )

    print("atomcoords_O_out:", atomcoords_O_out.tolist())
    print("H2_candidates[0,0] (O1 side):", H2[0, 0].tolist())
    print("H2_candidates[0,1] (O2 side):", H2[0, 1].tolist())



