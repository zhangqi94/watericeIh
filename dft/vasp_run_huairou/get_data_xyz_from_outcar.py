import os
from ase.io.vasp import read_vasp_out
from ase.io import write
import random

####################################################################################################

def get_data_from_outcar(root_folders, output_xyz):
    
    os.makedirs(os.path.dirname(output_xyz), exist_ok=True)
    # Initialize statistics dictionary to track number of OUTCAR files and total frames per folder
    folder_stats = {folder: {'num_outcars': 0, 'total_frames': 0} for folder in root_folders}

    counts_num =0
    # Open the output file in write mode
    with open(output_xyz, "w") as outfile:
        for root_folder in root_folders:
            for dirpath, _, filenames in os.walk(root_folder):
                if "OUTCAR" in filenames: 
                    outcar_path = os.path.join(dirpath, "OUTCAR")
                    try:
                        # Read all frames from the OUTCAR file
                        # `:` means all frames
                        atoms_list = read_vasp_out(outcar_path, index=":")
                        num_frames = len(atoms_list)
                        folder_stats[root_folder]['num_outcars'] += 1
                        folder_stats[root_folder]['total_frames'] += num_frames
                        
                        for atoms in atoms_list:
                            # Extract necessary information from each frame
                            energy = atoms.get_potential_energy()  # Potential energy
                            forces = atoms.get_forces()  # Atomic forces
                            lattice = atoms.cell.array  # Lattice matrix
                            symbols = atoms.get_chemical_symbols()  # Atomic symbols
                            positions = atoms.get_positions()  # Atomic positions
                            charges = atoms.get_atomic_numbers()  # Atomic charges

                            # Write the header for the frame
                            outfile.write(f"{len(atoms)}\n")
                            outfile.write(
                                f"TotEnergy={energy:.8f} "
                                f'pbc="T T T" Lattice="{lattice[0,0]:.8f} {lattice[0,1]:.8f} {lattice[0,2]:.8f} '
                                f'{lattice[1,0]:.8f} {lattice[1,1]:.8f} {lattice[1,2]:.8f} '
                                f'{lattice[2,0]:.8f} {lattice[2,1]:.8f} {lattice[2,2]:.8f}" '
                                f'Properties=species:S:1:pos:R:3:force:R:3:Z:I:1\n'
                            )

                            # Write atomic data (symbols, positions, forces, and charges)
                            for symbol, pos, force, charge in zip(symbols, positions, forces, charges):
                                outfile.write(
                                    f"{symbol:<2} {pos[0]:>15.8f} {pos[1]:>15.8f} {pos[2]:>15.8f} "
                                    f"{force[0]:>15.8f} {force[1]:>15.8f} {force[2]:>15.8f} {charge:>8d}\n"
                                )
                        
                        counts_num += 1
                        print(f"Counts: {counts_num}  Processed: {outcar_path} ({len(atoms_list)} frames)")
                    except Exception as e:
                        print(f"Failed to process {outcar_path}: {e}")

    for folder, stats in folder_stats.items():
        print(f"Folder: {folder}")
        print(f"  Number of OUTCAR files: {stats['num_outcars']}")
        print(f"  Total number of frames: {stats['total_frames']}")

        print(f"All data combined into {output_xyz}")

    return folder_stats

####################################################################################################
def split_xyz_file(input_file, train_file, test_file, train_ratio=0.8):
    """
    Randomly splits a .xyz file into train and test sets
    Arguments:
        input_file: Path to the input .xyz file
        train_file: Path to the output train .xyz file
        test_file: Path to the output test .xyz file
        train_ratio: Proportion of data to be used for training (0~1, default is 0.8)
    """
    with open(input_file, "r") as infile:
        lines = infile.readlines()
    
    # Determine the number of lines per frame
    frames = []
    i = 0
    while i < len(lines):
        natoms = int(lines[i].strip())  # The first line of each frame is the number of atoms
        frame = lines[i:i + natoms + 2]  # Includes the atom count line and the comment line
        frames.append(frame)
        i += natoms + 2
    
    # Shuffle frames randomly
    random.shuffle(frames)
    
    # Split frames based on the given ratio
    train_size = int(len(frames) * train_ratio)
    train_frames = frames[:train_size]
    test_frames = frames[train_size:]
    
    # Write train and test data to their respective files
    with open(train_file, "w") as train_out:
        for frame in train_frames:
            train_out.writelines(frame)
    
    with open(test_file, "w") as test_out:
        for frame in test_frames:
            test_out.writelines(frame)
    
    print(f"Dataset split complete: {len(train_frames)} frames in train, {len(test_frames)} frames in test")
    
####################################################################################################
if __name__ == '__main__':
    
    ### all datas from vasp
    root_folders = [
        "/home/users/zhangqi/zqcode/watericeIh_vasp/datavasp/mc_ff_n032",
        "/home/users/zhangqi/zqcode/watericeIh_vasp/datavasp/mc_ff_n064",
        "/home/users/zhangqi/zqcode/watericeIh_vasp/datavasp/mc_ff_n096",
        "/home/users/zhangqi/zqcode/watericeIh_vasp/datavasp/mc_ff_n128",
        "/home/users/zhangqi/zqcode/watericeIh_vasp/datavasp/mc_n016",
        "/home/users/zhangqi/zqcode/watericeIh_vasp/datavasp/mc_n032",
        "/home/users/zhangqi/zqcode/watericeIh_vasp/datavasp/mc_n064",
        "/home/users/zhangqi/zqcode/watericeIh_vasp/datavasp/mc_n096",
        "/home/users/zhangqi/zqcode/watericeIh_vasp/datavasp/mc_n128",
    ]
    output_xyz = "/home/users/zhangqi/zqcode/watericeIh_vasp/data_xyz_251112/watericeIh.xyz"
    folder_stats = get_data_from_outcar(root_folders, output_xyz)
    
    ### split data into train and test

    input_xyz = "/home/users/zhangqi/zqcode/watericeIh_vasp/data_xyz_251112/watericeIh.xyz"
    train_xyz = "/home/users/zhangqi/zqcode/watericeIh_vasp/data_xyz_251112/watericeIh_train.xyz"
    test_xyz  = "/home/users/zhangqi/zqcode/watericeIh_vasp/data_xyz_251112/watericeIh_test.xyz"

    split_xyz_file(input_xyz, train_xyz, test_xyz, train_ratio=0.90)
    
"""
conda activate numpy
cd /home/users/zhangqi/zqcode/watericeIh_vasp
python3 get_data_xyz_from_outcar.py
"""
