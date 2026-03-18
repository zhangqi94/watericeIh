"""Physical constants and unit conversions for molecular simulations.

All constants use CONSTANT_CASE naming convention as per PEP 8.
Units are organized by category for easy reference.
"""

####################################################################################################
# Fundamental Physical Constants
####################################################################################################

# Boltzmann constant
K_B_EV_PER_K = 8.617333262145e-5  # eV/K
K_B_J_PER_K = 1.380649e-23  # J/K

# Avogadro constant
N_A = 6.02214076e23  # mol^-1

# Elementary charge
E_CHARGE = 1.602176634e-19  # Coulomb

# Speed of light
C_LIGHT = 299792458.0  # m/s

# Planck constant
H_PLANCK = 6.62607015e-34  # J*s
H_PLANCK_EV = 4.135667696e-15  # eV*s
HBAR = H_PLANCK / (2.0 * 3.141592653589793)  # J*s
HBAR_EV = H_PLANCK_EV / (2.0 * 3.141592653589793)  # eV*s

####################################################################################################
# Atomic Masses (amu)
####################################################################################################

# Hydrogen isotopes
MASS_H1 = 1.007825031898  # amu (protium)
MASS_H2 = 2.014101777844  # amu (deuterium)
MASS_H3 = 3.016049281320  # amu (tritium)
MASS_H = MASS_H1  # Default hydrogen mass

# Oxygen isotopes
MASS_O16 = 15.994914619257  # amu
MASS_O17 = 16.999131755953  # amu
MASS_O18 = 17.999159612136  # amu
MASS_O = MASS_O16  # Default oxygen mass

# Water molecule
MASS_H2O = 2.0 * MASS_H + MASS_O  # amu (approximately 18.015)

####################################################################################################
# Mass Unit Conversions
####################################################################################################

AMU_TO_KG = 1.660539066e-27  # kg
AMU_TO_G = 1.660539066e-24  # g

####################################################################################################
# Energy Unit Conversions
####################################################################################################

# eV conversions
EV_TO_J = 1.602176634e-19  # J
J_TO_EV = 1.0 / EV_TO_J  # eV
EV_TO_KCAL_MOL = 23.060548  # kcal/mol
EV_TO_KJ_MOL = 96.485332  # kJ/mol
MEV_TO_EV = 1.0e-3  # eV
EV_TO_MEV = 1.0e3  # meV

# Hartree (atomic unit of energy)
HARTREE_TO_EV = 27.211386245988  # eV
EV_TO_HARTREE = 1.0 / HARTREE_TO_EV

####################################################################################################
# Length Unit Conversions
####################################################################################################

# Angstrom conversions
ANGSTROM_TO_M = 1.0e-10  # m
M_TO_ANGSTROM = 1.0e10  # Angstrom
ANGSTROM_TO_CM = 1.0e-8  # cm
CM_TO_ANGSTROM = 1.0e8  # Angstrom
ANGSTROM_TO_NM = 0.1  # nm
NM_TO_ANGSTROM = 10.0  # Angstrom

# Bohr radius (atomic unit of length)
BOHR_TO_ANGSTROM = 0.529177210903  # Angstrom
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM

####################################################################################################
# Volume Unit Conversions
####################################################################################################

ANGSTROM3_TO_M3 = 1.0e-30  # m^3
M3_TO_ANGSTROM3 = 1.0e30  # Angstrom^3
ANGSTROM3_TO_CM3 = 1.0e-24  # cm^3
CM3_TO_ANGSTROM3 = 1.0e24  # Angstrom^3
ANGSTROM3_TO_NM3 = 1.0e-3  # nm^3
NM3_TO_ANGSTROM3 = 1.0e3  # Angstrom^3

####################################################################################################
# Pressure Unit Conversions
####################################################################################################

# Pressure conversions
GPA_TO_PA = 1.0e9  # Pa
PA_TO_GPA = 1.0e-9  # GPa
GPA_TO_EV_PER_ANGSTROM3 = 0.00624150913  # eV/Angstrom^3
EV_PER_ANGSTROM3_TO_GPA = 1.0 / GPA_TO_EV_PER_ANGSTROM3

ATM_TO_PA = 101325.0  # Pa
PA_TO_ATM = 1.0 / ATM_TO_PA
ATM_TO_GPA = ATM_TO_PA * PA_TO_GPA  # approximately 0.000101325 GPa
GPA_TO_ATM = 1.0 / ATM_TO_GPA

BAR_TO_PA = 1.0e5  # Pa
PA_TO_BAR = 1.0e-5  # bar
BAR_TO_GPA = BAR_TO_PA * PA_TO_GPA  # 1e-4 GPa
GPA_TO_BAR = 1.0 / BAR_TO_GPA

####################################################################################################
# Time Unit Conversions
####################################################################################################

FS_TO_S = 1.0e-15  # s (femtosecond)
PS_TO_S = 1.0e-12  # s (picosecond)
NS_TO_S = 1.0e-9  # s (nanosecond)
S_TO_FS = 1.0e15  # fs
S_TO_PS = 1.0e12  # ps
S_TO_NS = 1.0e9  # ns

####################################################################################################
# Force Unit Conversions
####################################################################################################

# Force conversions (eV/Angstrom is common in MD simulations)
EV_PER_ANGSTROM_TO_N = EV_TO_J / ANGSTROM_TO_M  # N
N_TO_EV_PER_ANGSTROM = 1.0 / EV_PER_ANGSTROM_TO_N

####################################################################################################
# Dipole Moment Conversions
####################################################################################################

DEBYE_TO_C_M = 3.33564e-30  # C*m
C_M_TO_DEBYE = 1.0 / DEBYE_TO_C_M
DEBYE_TO_E_ANGSTROM = 0.2081943  # e*Angstrom
E_ANGSTROM_TO_DEBYE = 1.0 / DEBYE_TO_E_ANGSTROM

####################################################################################################
# Temperature Conversion Functions
####################################################################################################

def celsius_to_kelvin(celsius: float) -> float:
    """Convert Celsius to Kelvin.

    Args:
        celsius: Temperature in Celsius

    Returns:
        Temperature in Kelvin
    """
    return celsius + 273.15


def kelvin_to_celsius(kelvin: float) -> float:
    """Convert Kelvin to Celsius.

    Args:
        kelvin: Temperature in Kelvin

    Returns:
        Temperature in Celsius
    """
    return kelvin - 273.15


def kelvin_to_ev(kelvin: float) -> float:
    """Convert temperature in Kelvin to thermal energy in eV.

    Args:
        kelvin: Temperature in Kelvin

    Returns:
        Thermal energy k_B*T in eV
    """
    return K_B_EV_PER_K * kelvin


def ev_to_kelvin(ev: float) -> float:
    """Convert thermal energy in eV to temperature in Kelvin.

    Args:
        ev: Thermal energy in eV

    Returns:
        Temperature in Kelvin
    """
    return ev / K_B_EV_PER_K


####################################################################################################
# Density Calculation Functions
####################################################################################################

def calculate_density_g_per_cm3(mass_g: float, volume_angstrom3: float) -> float:
    """Calculate density in g/cm^3 from mass and volume.

    Args:
        mass_g: Total mass in grams
        volume_angstrom3: Volume in Angstrom^3

    Returns:
        Density in g/cm^3
    """
    volume_cm3 = volume_angstrom3 * ANGSTROM3_TO_CM3
    return mass_g / volume_cm3


def calculate_mass_h2o_g(num_molecules: int) -> float:
    """Calculate total mass of water molecules in grams.

    Args:
        num_molecules: Number of H2O molecules

    Returns:
        Total mass in grams
    """
    return num_molecules * MASS_H2O * AMU_TO_G


def calculate_number_density(num_molecules: int, volume_angstrom3: float) -> float:
    """Calculate number density in molecules/Angstrom^3.

    Args:
        num_molecules: Number of molecules
        volume_angstrom3: Volume in Angstrom^3

    Returns:
        Number density in molecules/Angstrom^3
    """
    return num_molecules / volume_angstrom3


####################################################################################################
# Example Usage and Tests
####################################################################################################

if __name__ == "__main__":
    print("=" * 80)
    print("Physical Constants and Unit Conversions")
    print("=" * 80)

    # Fundamental constants
    print("\n[Fundamental Constants]")
    print(f"  Boltzmann constant: {K_B_EV_PER_K:.6e} eV/K")
    print(f"  Avogadro constant:  {N_A:.6e} mol^-1")

    # Atomic masses
    print("\n[Atomic Masses]")
    print(f"  H (protium):  {MASS_H:.6f} amu")
    print(f"  O-16:         {MASS_O:.6f} amu")
    print(f"  H2O:          {MASS_H2O:.6f} amu")

    # Temperature conversion
    print("\n[Temperature Conversions]")
    T_K = 300.0
    T_eV = kelvin_to_ev(T_K)
    print(f"  {T_K:.1f} K = {T_eV:.6f} eV = {T_eV * EV_TO_MEV:.3f} meV")

    # Pressure conversion
    print("\n[Pressure Conversions]")
    P_atm = 1.0
    P_GPa = P_atm * ATM_TO_GPA
    P_eV_A3 = P_GPa * GPA_TO_EV_PER_ANGSTROM3
    print(f"  {P_atm:.1f} atm = {P_GPa:.6f} GPa = {P_eV_A3:.9f} eV/Angstrom^3")

    # Density calculation
    print("\n[Density Calculation Example]")
    num_H2O = 96
    volume_A3 = 3000.0
    mass_g = calculate_mass_h2o_g(num_H2O)
    density = calculate_density_g_per_cm3(mass_g, volume_A3)
    print(f"  {num_H2O} H2O molecules in {volume_A3:.1f} Angstrom^3")
    print(f"  Mass: {mass_g:.6e} g")
    print(f"  Density: {density:.4f} g/cm^3")

    # Energy conversions
    print("\n[Energy Conversions]")
    E_eV = 1.0
    print(f"  {E_eV:.1f} eV = {E_eV * EV_TO_KCAL_MOL:.3f} kcal/mol")
    print(f"  {E_eV:.1f} eV = {E_eV * EV_TO_KJ_MOL:.3f} kJ/mol")
    print(f"  {E_eV:.1f} eV = {E_eV * EV_TO_MEV:.1f} meV")

    print("\n" + "=" * 80)
