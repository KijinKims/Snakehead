import sys
from typing import NewType, Set, Tuple, List
from collections import defaultdict, deque
from pathlib import Path, PurePath
import copy
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from hmm_profile.reader import read_single
from hmm_profile.models import HMM
import graph_tool as gt
import graph_tool.topology
import graph_tool.draw
import profileHMM
import time
import argparse
import pyhmmer
from functools import wraps

Idx = NewType('Idx', int)
HMMId = NewType('HMMId', str)
ContigId = NewType('ContigId', str)
Strand = NewType('Strand', str)

def timeit(func):
    """Measure the time lapsed"""
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} took {total_time:.1f} seconds')
        return result
    return timeit_wrapper

class DAG:
    """The class encapsulating Networkx Digraph object.

    Attributes:
        _name: A string indicating the name of the graph.
        _graph: A Networkx Digraph object.
        _ordering: A list of strings that are node ids of the graph.
        _len: An integer count of the nodes in the graph.
        _source: A string indicating the node id of source node in the graph.
        _sink: A string indicating the node id of sink node in the graph.
        _node_id_to_index_dict: A dictionary matching every node id to index in self._ordering list. It is updated every time ordering is updated.
    """

    def __init__(self, graph_ : gt.Graph):
        """Inits DAG with a given Networkx Digraph object."""
        self._name = graph_.graph_properties['id']
        self._graph : gt.Graph = graph_
        self._ordering = graph_tool.topology.topological_sort(self._graph)
        self._len : int = self._graph.num_vertices()

        for v in self._graph.vertices():
            if v.in_degree() == 0:
                self._source = int(v)

        for v in self._graph.vertices():
            if v.out_degree() == 0:
                self._sink = int(v)

    def __len__(self) -> int :
        """Return the number of nodes in the graph."""
        return self._len

    def len(self, len_ : int) :
        """Set the number of nodes in the graph."""
        assert len_ > 0
        self._len = len_

    @property
    def name(self):
        """Return the name of the graph."""
        return self._name

    @property
    def ordering(self):
        """Return the sorted order of nodes in the graph. Typically, they are topologically sorted."""
        return self._ordering

    @ordering.setter
    def ordering(self, ordering_):
        """Set orted order of nodes in the graph. Typically, they are topologically sorted. Afterwards, update self._node_id_to_index_dict."""
        self._ordering = ordering_
        self.set_node_id_to_index()

    @timeit
    def update_ordering(self):
        """Update ordering"""
        self.ordering = graph_tool.topology.topological_sort(self._graph)

    def predecessors(self, node_id_):
        return self._graph.get_in_neighbors(node_id_)

    def set_node_id_to_index(self) :
        """Indexize node ids to index in self._ordering in order to make them fit to numpy operations."""
        self._node_id_to_index_dict = { val : Idx(idx) for idx, val in enumerate(self._ordering) }

    def node_id_to_index(self, node_id_) -> Idx :
        """Return index in self._ordering of the node with given node id."""
        return self._node_id_to_index_dict[node_id_]

    def index_to_node_id(self, idx_ : Idx):
        """Return the node id corresponding given index in self._ordering."""
        return self._ordering[idx_]

    
    @timeit
    def update_base_dict(self) :
        self._base_dict = dict()
        for v in self._graph.vertices():
            self._base_dict[int(v)] = self._graph.vertex_properties['base'][v]

    def base(self, node_id_):
        """Return the base of the node with given node id."""
        return self._base_dict[node_id_]

    def reverse(self):
        """Reverse the graph."""
        self._graph.set_reversed(True)

    def complement(self):
        """Change each base of nodes in the graph as complimentary base."""
        complement_dict = { "A": "T",
                            "T": "A",
                            "G": "C",
                            "C": "G",
                            } 
        
        prop = self._graph.new_vertex_property("string")
        for v in self._graph.vertices():
            original = self._graph.vertex_properties['base'][v]

            if original in ["^", "$"]:
                prop[v] = original
            
            else:
                prop[v] = complement_dict[original]
        self._graph.vertex_properties['base'] = prop

    def swap_source_sink(self):
        """Swap source and sink node."""
        self._source, self._sink = self._sink, self._source

    @timeit
    def reverse_complement(self):
        """Reverse and complement the graph and swap source and sink to make it plausible."""
        self.reverse()
        self.complement()
        self.swap_source_sink()

    def extract_seq_from_path(self, node_id_list_) -> str:
        """Return the sequence concatenating bases extracted from each node with the node id in given list."""
        sequence : str = ""
        for node_id in node_id_list_:
            if self.base(node_id) == "^" or self.base(node_id) == "$":
                continue
            sequence += self.base(node_id)
        return sequence

@timeit
def split_graphs(graphml_path_ : Path, contigids : List[ContigId], tmpdir : Path) :
    """Split graphs into single DAGs"""

    contigid : ContigId = ""
    single_graph=""
    
    with open(graphml_path_) as graphml:
        for line in graphml:
            if line.startswith('<?xml'): # start of new graph

                if single_graph and contigid in contigids:
                    with open(Path(tmpdir, f"{contigid}.graphml"), "w") as tmp:
                        tmp.write(single_graph)
                        contigid = ""
                
                single_graph = line

            else: 
                single_graph += line
                if line.startswith('  <graph'):
                    contigid = line.split()[1].split('"')[1]
                
    
    # last graph
    if contigid in contigids:
        with open(Path(tmpdir, f"{contigid}.graphml"), "w") as tmp:
            tmp.write(single_graph)
            
@timeit
def read_graphml(tmpdir : Path, consensus_id) -> DAG :
    graphml = PurePath(tmpdir, f"{consensus_id}.graphml")
    
    dag = DAG(gt.load_graph(str(graphml)))

    return dag

@timeit
def fetch_hmm(hmm_path_ : Path) :
    """fetch hmm from profile hmm file."""

    with open(hmm_path_, "r") as f:
        model = read_single(f)

    return model

@timeit
def parse_nhmmer_result(hmmscan_tbl_path_ : Path):
    """Parse nhmmer tblout file"""
    hits = set()
    with open(hmmscan_tbl_path_) as f:
        for line in f:
            if line.startswith('#'):
                continue
            else:
                li = line.split()
                target_name = li[0]
                query_name = li[2]
                strand = li[11]
                hits.add((target_name, query_name, strand))
    return hits

@timeit
def consensus(dag_ : DAG, phmm_ : HMM):
    """Correct the path in DAG using Viterbi algorithm."""
    PHMM = profileHMM.PHMM(phmm_)
    path = PHMM.viterbi(dag_)

    return path

def sequence_as_seqrecord_with_ids(sequence_ : str, consensus_id_, phmm_id_):
    """Write the corrected consensus as a FASTA format with the header including consensus ID and HMM ID used for correction."""
    seqid = f'{consensus_id_}_corrected_with_{phmm_id_}'
    record = SeqRecord(
        Seq(sequence_),
        id=seqid,
        description=f'{consensus_id_} len={len(sequence_)}')

    return record

parser = argparse.ArgumentParser(prog='Snakehead', description='%(prog)s is a command line program for correcting sequencing errors in Nanopore contigs with pHMM.')
parser.add_argument('--prefix', '-p', nargs='?')

# input files
parser.add_argument('--tbl', '-c', nargs='?')
parser.add_argument('--graphs', '-g', nargs='?', help="DAG made by Canu in graphml format.")
parser.add_argument('--hmm', nargs='?', help="profile HMM file.")

# output files
parser.add_argument('--outdir', '-o', nargs='?', help="Directory that output files will be stored.")
parser.add_argument('--tmpdir', nargs='?', default='tmp', help="Directory that temporary files will be stored.")
parser.add_argument('--log', nargs='?')

# parameters
parser.add_argument('--skip_split_graphs', action='store_true', default=False, help="For less memory use, graphs are stored in 'tempdir' one by one. By turning on this flag, this process can be skipped if it was done before.")

args = parser.parse_args()

if __name__ == '__main__':
    bb_start = time.perf_counter()
    print(f"Start the correction for prefix {args.prefix}.")

    hmmscan_tbl_file : Path = Path(args.tbl)
    graphs_file : Path = Path(args.graphs)
    hmm_file : Path = Path(args.hmm)
    outdir : Path = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tmpdir : Path = Path(args.tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)

    if args.log:
        log = open(args.log, "a")

    hmmscan_hits = parse_nhmmer_result(hmmscan_tbl_file)
    if len(hmmscan_hits) == 0:
        sys.exit("No hits found between given profile HMM and contigs.")

    hmmid : HMMId
    contigid : ContigId
    strand : Strand # if alignment is in reverse order, DAG also should be reversed.

    # extract contig IDs
    contigids : List[ContigId] = []
    for hit in hmmscan_hits:
        contigid, hmmid, strand = hit
        contigids.append(contigid)

    # split graphs one by one for less memory use. each graph is stored in 'tempdir' with {graph id}.graphml as file name.
    if not args.skip_split_graphs:
        split_graphs(graphs_file, contigids, tmpdir)


    # profileHMM._viterbi() is compiled by Numba's @JIT(Just-In-Time) decorator
    # As Numba supports caching the compilation for reuse,
    # correct() function is called only for caching.
    contigid, hmmid, strand = list(hmmscan_hits)[0]
    graph = read_graphml(tmpdir, contigid)
    graph.update_ordering()
    graph.update_base_dict()
    hmm = fetch_hmm(hmm_file)
    corr_path = consensus(graph, hmm)

    # Actual corrections start from here.
    for hit in hmmscan_hits:
        contigid, hmmid, strand = hit

        b_start = time.perf_counter()
        print(f"Start the correction. Contig {contigid} with pHMM {hmmid}.")

        ## load dag
        graph = read_graphml(tmpdir, contigid)

        ## set up dag
        # if reverse hit, reversed dag needed
        if strand == '-':
            graph.reverse_complement()

        graph.update_ordering()
        graph.update_base_dict()

        ## fetch hmm and correct
        hmm = fetch_hmm(hmm_file)
        corr_path = consensus(graph, hmm)
        corr_seq = graph.extract_seq_from_path(corr_path)

        ## export results
        SeqIO.write(sequence_as_seqrecord_with_ids(corr_seq, contigid, hmmid), Path(outdir, f"{args.prefix}_{contigid}_{hmmid}.fasta"), "fasta")

        b_end = time.perf_counter()
        print(f"End the correction. Contig {contigid} with pHMM {hmmid}. Elapsed time(sec):", b_end - b_start)

        if args.log:
            log.write(f"{args.prefix}\t{contigid}\t{hmmid}\t{b_end - b_start}\n")

    if args.log:
        log.close()
    bb_end = time.perf_counter()
    print(f"End the correction for prefix {args.prefix}. Elapsed time(sec):", bb_end - bb_start)
