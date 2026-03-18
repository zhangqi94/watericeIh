#%%
####################################################################################################
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

npz_file = "/home/zq/zqcodeml/watericeIh_data/dft_data/data_xyz_260201/mace_iceIh_128x0e128x1o_r4.5_float32_seed146_testset_gpu_v2.npz"


data = np.load(npz_file, allow_pickle=True)

# ======================
# Load data
# ======================
E_ref = data["E_ref"]
E_mace = data["E_mace"]
F_ref = data["F_ref"]
F_mace = data["F_mace"]
n_atoms = data["n_atoms"]

# ---------- Energy per H2O ----------
n_atoms = data["n_atoms"].astype(int)
n_h2o = (n_atoms // 3).astype(float)

E_ref_h2o  = E_ref  / n_h2o
E_mace_h2o = E_mace / n_h2o

mask_E = np.isfinite(E_ref_h2o) & np.isfinite(E_mace_h2o)
E_ref_h2o  = E_ref_h2o[mask_E]
E_mace_h2o = E_mace_h2o[mask_E]
dE_meV = (E_mace_h2o - E_ref_h2o) * 1000.0  # meV/H2O

# ---------- Forces ----------
# flatten all forces into 1D float array
F_ref_flat  = np.concatenate([f.reshape(-1) for f in F_ref]).astype(float)
F_mace_flat = np.concatenate([f.reshape(-1) for f in F_mace]).astype(float)
dF_meV = (F_mace_flat - F_ref_flat) * 1000.0  # meV/Å

# ---------- histogram controls ----------
E_hist_bins  = 80
E_hist_range = (-1.5, 1.5)      # meV/H2O, 你自己改

F_hist_bins  = 120
F_hist_range = (-30.0, 30.0)    # meV/Å, 你自己改

# ======================
# Metrics
# ======================
rmse_E = np.sqrt(np.mean(dE_meV**2))  # meV/H2O
rmse_F = np.sqrt(np.mean(dF_meV**2))  # meV/Å

# ======================
# Plot
# ======================
fig, axes = plt.subplots(1, 2, figsize=(12.0, 6.0), dpi=300)

# ------------------------------------------------------------------
# (a) Energy parity
# ------------------------------------------------------------------
ax = axes[0]

ax.scatter(E_ref_h2o, E_mace_h2o, s=5, alpha=0.8)

emin, emax = -14.90, -14.78
ax.plot([emin, emax], [emin, emax], "r--", lw=1.5)

ax.set_xlim(emin, emax)
ax.set_ylim(emin, emax)
ax.set_xlabel("DFT energy (eV/H$_2$O)", fontsize=14)
ax.set_ylabel("MACE energy (eV/H$_2$O)", fontsize=14)

# (a) bold + RMSE moved here
ax.text(
    0.03, 0.97,
    rf"$\mathbf{{a}}$  RMSE = {rmse_E:.2f} meV/H$_2$O",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=14,
)

# inset: energy residual
axins = inset_axes(
    ax,
    width="38%",
    height="38%",
    loc="lower right",
    bbox_to_anchor=(-0.02, 0.12, 1, 1),  # 往上挪
    bbox_transform=ax.transAxes,
    borderpad=0.0,
)
axins.hist(dE_meV, bins=E_hist_bins, range=E_hist_range, histtype="stepfilled")
axins.set_xlabel(r"$E_{\mathrm{MACE}}-E_{\mathrm{DFT}}$ (meV/H$_2$O)", fontsize=12)
axins.set_ylabel("Counts", fontsize=12)
axins.tick_params(labelsize=10)

# ------------------------------------------------------------------
# (b) Force parity
# ------------------------------------------------------------------
ax = axes[1]

ax.scatter(F_ref_flat, F_mace_flat, s=5, alpha=0.5)

fmin, fmax = -4.0, 4.0
ax.plot([fmin, fmax], [fmin, fmax], "r--", lw=1.5)

ax.set_xlim(fmin, fmax)
ax.set_ylim(fmin, fmax)
ax.set_xlabel("DFT force (eV/Å)", fontsize=14)
ax.set_ylabel("MACE force (eV/Å)", fontsize=14)

# (b) bold + RMSE moved here
ax.text(
    0.03, 0.97,
    rf"$\mathbf{{b}}$  RMSE = {rmse_F:.1f} meV/Å",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=14,
)

# inset: force residual
axins = inset_axes(
    ax,
    width="38%",
    height="38%",
    loc="lower right",
    bbox_to_anchor=(-0.02, 0.12, 1, 1),  # 往上挪
    bbox_transform=ax.transAxes,
    borderpad=0.0,
)
axins.hist(dF_meV, bins=F_hist_bins, range=F_hist_range, histtype="stepfilled")
axins.set_xlabel(r"$F_{\mathrm{MACE}}-F_{\mathrm{DFT}}$ (meV/Å)", fontsize=12)
axins.set_ylabel("Counts", fontsize=12)
axins.tick_params(labelsize=10)

# ======================
plt.tight_layout()
plt.show()