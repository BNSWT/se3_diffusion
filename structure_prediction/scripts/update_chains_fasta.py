# this code is made for updating OpenProteinSet's training data to the new mmCIF databse
# use obsolete file to update all old chain name to new chain name if there exist a mapping
from Bio import SeqIO
import argparse

parser = argparse.ArgumentParser(description='''this code is made for updating 
    OpenProteinSet's training data to the new mmCIF databse use obsolete file, 
    update all old chain name to new chain name if there exist a mapping,else
    remove this chain from training set by dump chains.list, which will be use
    by dataset preprocess''')

parser.add_argument("--chain_list",type=str,help="the file contains the chain name of each chain of OpenProteinSet,can get by (ls pdb >> chains.list)")
parser.add_argument("--pdbseq_res",type=str,help="the pdb_seqres.txt of pdb database")
parser.add_argument("--obsolete",type=str,help="the obsolete.dat of pdb database")
parser.add_argument("--output_fasta",type=str,help="the output fasta list",default="chains.fasta")
parser.add_argument("--output_chains",type=str,help="the output fasta list",default="chains.list")


def main(args):
    records =  SeqIO.parse(args.pdbseq_res,"fasta")
    fasta_dict = {record.description.split()[0]:record.seq for record in records}
    chains = open(args.chain_list).read().strip().split()
    obsolete = {line.split()[2].lower():line.split()[3].lower() for line in open(args.obsolete).readlines()[1:] if line.split().__len__()==4}
    chains = []
    with open(args.output_fasta,"w") as f:
        for i in chains:
            try:
                chain = i if i[:4] not in obsolete else obsolete[i[:4]]+'_'+i[5:]
                seq = fasta_dict[chain]
                chains.append(chain)
                f.write(f">{chain}\n{seq}\n")
            except:
                continue
    open(args.output_chains).write("\n".join(chains))

if __name__ == "main":
    args = parser.parse_args()
    main(args)