import sys
import io as sysio

import numpy as np
import pyrosetta
from pyrosetta import *
from pyrosetta.rosetta import *

    
def extract_coords_from_pose(pose, atoms=['N','CA','C'], chain=None):
  '''
  input:  x = PDB filename
          atoms = atoms to extract (optional)
  output: (length, atoms, coords=(x,y,z)), sequence
  '''

  alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
  states = len(alpha_1)
  alpha_3 = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
             'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','GAP']
  
  aa_1_N = {a:n for n,a in enumerate(alpha_1)}
  aa_3_N = {a:n for n,a in enumerate(alpha_3)}
  aa_N_1 = {n:a for n,a in enumerate(alpha_1)}
  aa_1_3 = {a:b for a,b in zip(alpha_1,alpha_3)}
  aa_3_1 = {b:a for a,b in zip(alpha_1,alpha_3)}
  
  def AA_to_N(x):
    # ["ARND"] -> [[0,1,2,3]]
    x = np.array(x);
    if x.ndim == 0: x = x[None]
    return [[aa_1_N.get(a, states-1) for a in y] for y in x]
  
  def N_to_AA(x):
    # [[0,1,2,3]] -> ["ARND"]
    x = np.array(x);
    if x.ndim == 1: x = x[None]
    return ["".join([aa_N_1.get(a,"-") for a in y]) for y in x]

  def rosetta_xyz_to_numpy(x):
    return np.array([x.x, x.y, x.z])

  xyz,seq,min_resn,max_resn = {},{},1e6,-1e6

  # the pdb info struct
  info = pose.pdb_info()

  for resi in range(1, pose.size()+1):
    # for each residue
    ch = info.chain(resi)

    if ch == chain:
        residue = pose.residue(resi)

        resn = resi
        resa,resn = "",int(resn)-1

        if resn < min_resn: 
            min_resn = resn
        if resn > max_resn: 
            max_resn = resn

        xyz[resn] = {}
        xyz[resn][resa] = {}

        seq[resn] = {}
        seq[resn][resa] = residue.name3()

        # for each heavy atom
        for iatm in range(1, residue.nheavyatoms()+1):
            atom_name = residue.atom_name(iatm).strip()
            atom_xyz = rosetta_xyz_to_numpy( residue.xyz(iatm) )

            xyz[resn][resa][atom_name] = atom_xyz

  # convert to numpy arrays, fill in missing values
  seq_,xyz_ = [],[]
  try:
      for resn in range(min_resn,max_resn+1):
        if resn in seq:
          for k in sorted(seq[resn]): seq_.append(aa_3_N.get(seq[resn][k],20))
        else: seq_.append(20)
        if resn in xyz:
          for k in sorted(xyz[resn]):
            for atom in atoms:
              if atom in xyz[resn][k]: xyz_.append(xyz[resn][k][atom])
              else: xyz_.append(np.full(3,np.nan))
        else:
          for atom in atoms: xyz_.append(np.full(3,np.nan))
      return np.array(xyz_).reshape(-1,len(atoms),3), N_to_AA(np.array(seq_))
  except TypeError:
      return 'no_chain', 'no_chain'

def parse_pose(pose, use_chains,ca_only):
    c=0
    pdb_dict_list = []
    '''
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V','W','X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j','k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't','u', 'v','w','x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_alphabet = init_alphabet + extra_alphabet
    '''

    exist_chains = list(set([pose.pdb_info().chain(residue_id) for residue_id in range(1, pose.total_residue() + 1) ]))
    my_dict = {}
    s = 0
    concat_seq = ''
    for letter in exist_chains:
        if use_chains and letter not in use_chains:
          continue
        xyz, seq = extract_coords_from_pose(pose, atoms=['N','CA','C','O'] , chain=letter)
        if type(xyz) != str:
            concat_seq += seq[0]
            my_dict['seq_chain_'+letter]=seq[0]
            coords_dict_chain = {}
            if ca_only:
              coords_dict_chain['CA_chain_'+letter]=xyz[:,1,:].tolist()
            else:
              coords_dict_chain['N_chain_'+letter]=xyz[:,0,:].tolist()
              coords_dict_chain['CA_chain_'+letter]=xyz[:,1,:].tolist()
              coords_dict_chain['C_chain_'+letter]=xyz[:,2,:].tolist()
              coords_dict_chain['O_chain_'+letter]=xyz[:,3,:].tolist()
            my_dict['coords_chain_'+letter]=coords_dict_chain
            s += 1
    my_dict['name']=pose.pdb_info().name()
    my_dict['num_of_chains'] = s
    my_dict['seq'] = concat_seq
    pdb_dict_list.append(my_dict)
    return pdb_dict_list

def thread_mpnn_seq( pose, seq ):
    rsd_set = pose.residue_type_set_for_pose( core.chemical.FULL_ATOM_t )

    aa1to3=dict({'A':'ALA', 'C':'CYS', 'D':'ASP', 'E':'GLU', 'F':'PHE', 'G':'GLY',
        'H':'HIS', 'I':'ILE', 'K':'LYS', 'L':'LEU', 'M':'MET', 'N':'ASN', 'P':'PRO',
        'Q':'GLN', 'R':'ARG', 'S':'SER', 'T':'THR', 'V':'VAL', 'W':'TRP', 'Y':'TYR'})

    for resi, mut_to in enumerate( seq ):
        resi += 1 # 1 indexing
        name3 = aa1to3[ mut_to ]
        new_res = core.conformation.ResidueFactory.create_residue( rsd_set.name_map( name3 ) )
        pose.replace_residue( resi, new_res, True )
    
    return pose

def pose_to_string(pose):
  output = pyrosetta.rosetta.std.ostringstream()
  pose.dump_pdb(output)
  pose_str = output.str()
  return pose_str