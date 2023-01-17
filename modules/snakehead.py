from typing import NewType
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
from functools import wraps

Idx = NewType('Idx', int)

def timeit(func):
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
        _edge_threshold: An integer indicating the threshold of weight for filtering edges in the graph.
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

    @timeit
    def prun(self, edge_threshold_ : int) :
        """Remove the edges with weights lower than given threshold weight and update the graph and its attribute to make it reasonable.
        
        Initially, edges are filtered. It make some obsolete nodes, which has no zero in- and out-degree other than source and sink nodes. Such nodes are excluded from the graph.
        Update the self._len. Then, connect all zero indegree nodes other than source to source node by generating an edge between them. Likewise, connect all zero outdegree nodes other than sink to sink node.

        Args:
            edge_threshold_:
                An integer indicating the threshold of weight for filtering edges in the graph.
        """
        assert edge_threshold_ > 0
        self._edge_threshold = edge_threshold_
        self.filter_by_edge_weight(self._edge_threshold)
        #self.clean_obsolete_nodes()
        self.len(self._graph.num_vertices())
        self.leave_only_one_source()
        self.leave_only_one_sink()

    def filter_by_edge_weight(self, threshold_: int) :
        """Remove all edges with weight<threshold from the graph."""
        weight_filter = self._graph.new_edge_property("boolean", val=True)
        for e in self._graph.edges():
            if self._graph.edge_properties["weight"][e] < threshold_:
                weight_filter[e] = False
        
        self._graph.set_edge_filter(weight_filter)
        self._graph.purge_edges()
    
    def leave_only_one_source(self):
        """Connect all zero indegree nodes other than source to source"""
        sources = []
        for v in self._graph.vertices():
            if int(v) == self._source or int(v) == self._sink:
                continue
            if v.in_degree() == 0:
                sources.append(int(v))
        
        self._graph.add_edge_list([(self._source, x) for x in sources])

    def leave_only_one_sink(self):
        """Connect all zero outdegree nodes other than sink to sink"""
        sinks = []
        for v in self._graph.vertices():
            if int(v) == self._source or int(v) == self._sink:
                continue
            if v.out_degree() == 0:
                sinks.append(int(v))
        
        self._graph.add_edge_list([(x, self._sink) for x in sinks])

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

    def extract_sequence_from_path(self, node_id_list_) -> str:
        """Return the sequence concatenating bases extracted from each node with the node id in given list."""
        sequence : str = ""
        for node_id in node_id_list_:
            if self.base(node_id) == "^" or self.base(node_id) == "$":
                continue
            sequence += self.base(node_id)
        return sequence

@timeit
def split_graphs(graphml_path_ : Path, tigids, tmpdir : Path) :
    """Split graphs into single DAGs."""

    tigid = ""
    single_graph=""
    
    with open(graphml_path_) as graphml:
        for line in graphml:
            if line.startswith('<?xml'):

                if single_graph and tigid in tigids:
                    with open(Path(tmpdir, f"{tigid}.graphml"), "w") as tmp:
                        tmp.write(single_graph)
                        tigid = ""
                
                single_graph = line

            else: 
                single_graph += line
                if line.startswith('  <graph'):
                    tigid = line.split()[1].split('"')[1]
                
    
    # last graph
    if tigid in tigids:
        with open(Path(tmpdir, f"{tigid}.graphml"), "w") as tmp:
            tmp.write(single_graph)
            
@timeit
def read_graphml(tmpdir : Path, consensus_id) -> DAG :
    graphml = PurePath(tmpdir, f"{consensus_id}.graphml")
    
    dag = DAG(gt.load_graph(str(graphml)))

    return dag

@timeit
def fetch_hmm(hmmdb_path : Path, hmmid, tmpdir : Path) :
    """fetch hmm from hmm DB."""
    prev_line = deque(maxlen=1)
    write_on = False
    
    with open(Path(tmpdir, f"{hmmid}.hmm"), "w") as tmp:
        with open(hmmdb_path) as file:
            for line in file:
                
                if line.startswith(f'HMMER3'):
                    if write_on:
                        break

                if write_on:
                    tmp.write(line)
                
                if line.startswith(f'NAME  {hmmid}'):
                    tmp.write(prev_line.pop())
                    prev_line.clear()
                    tmp.write(line)
                    write_on = True
                
                prev_line.append(line)

    with open(Path(tmpdir, f"{hmmid}.hmm"), "r") as tmp:
        model = read_single(tmp)

    return model

@timeit
def parse_hmmscanresult(hmmscan_tbl_path_ : Path):
    """Parse nhmmscan tblout file"""
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
def correct(dag_ : DAG, phmm_ : HMM):
    """Correct the path in DAG using Viterbi algorithm."""
    PHMM = profileHMM.PHMM(phmm_)
    path = PHMM.viterbi(dag_)

    return path

def make_corrected_consensus_as_seqrecord(corrected_sequence_ : str, consensus_id_, phmm_id_):
    """Write the corrected consensus as a FASTA format with the header including consensus ID and HMM ID used for correction."""
    seqid = f'{consensus_id_}_corrected_with_{phmm_id_}'
    record = SeqRecord(
        Seq(corrected_sequence_),
        id=seqid,
        description=f'{consensus_id_} len={len(corrected_sequence_)}')

    return record

parser = argparse.ArgumentParser(prog='Snakehead', description='%(prog)s is a command line program for correcting sequencing errors in Nanopore contigs with pHMM.')
parser.add_argument('--prefix', '-p', nargs='?')
parser.add_argument('--graphs', '-g', nargs='?')
parser.add_argument('--outdir', '-o', nargs='?')
parser.add_argument('--tmpdir', nargs='?', default='tmp')
parser.add_argument('--tbl', '-d', nargs='?')
parser.add_argument('--hmm', nargs='?')
parser.add_argument('--log', nargs='?')
parser.add_argument('--minimum_edge_weight', type=int, default=1)
parser.add_argument('--skip_split_dot', action='store_true', default=False)

args = parser.parse_args()
#args = parser.parse_args(['--prefix', 'test_thrs_3', '--graphs', 'PR8_H1N1.graph.dot', '--outdir', 'test_thrs_3', '--tmpdir', 'tmp', '--domtbl', 'test.domtbl', '--hmm', '/home/kijin/DB/RVDB-prot/v23/U-RVDBv23.0-prot.hmm', '--minimum_edge_weight', '3'])


if __name__ == '__main__':
    bb_start = time.perf_counter()
    print(f"Start the correction for prefix {args.prefix}.")

    hmmscan_tbl_path : Path = Path(args.tbl)
    graphml_path : Path = Path(args.graphs)
    hmmdb_path : Path = Path(args.hmm)
    outdir : Path = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tmpdir : Path = Path(args.tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)
    minimum_edge_weight : int = args.minimum_edge_weight

    if args.log:
        log = open(args.log, "a")

    hmmscan_hits = parse_hmmscanresult(hmmscan_tbl_path)

    tigids = []
    for hit in hmmscan_hits:
        hmmid, tigid, strand = hit
        tigids.append(tigid)

    if not args.skip_split_dot:
        split_graphs(graphml_path, tigids, tmpdir)

    # for compiling _viterbi
    hmmid, tigid, strand = next(iter(hmmscan_hits))

    ## load dag
    dag = read_graphml(tmpdir, tigid)

    ## set up dag
    # if reverse hit, reversed dag needed
    if strand == '-':
        dag.reverse_complement()

    if minimum_edge_weight > 1:
        # filter low-weight edges
        dag.prun(minimum_edge_weight)
    
    dag.update_ordering()
    dag.update_base_dict()

    ## fetch hmm and correct
    hmm = fetch_hmm(hmmdb_path, hmmid, tmpdir)
    corrected_path = correct(dag, hmm)

    for hit in hmmscan_hits:
        hmmid, tigid, strand = hit

        b_start = time.perf_counter()
        print(f"Start the correction. Contig {tigid} with pHMM {hmmid}.")

        ## load dag
        dag = read_graphml(tmpdir, tigid)

        ## set up dag
        # if reverse hit, reversed dag needed
        if strand == '-':
            dag.reverse_complement()

        if minimum_edge_weight > 1:
            # filter low-weight edges
            dag.prun(minimum_edge_weight)
        
        dag.update_ordering()
        dag.update_base_dict()

        ## fetch hmm and correct
        hmm = fetch_hmm(hmmdb_path, hmmid, tmpdir)
        corrected_path = correct(dag, hmm)
        corrected_sequence = dag.extract_sequence_from_path(corrected_path)

        ## export results
        SeqIO.write(make_corrected_consensus_as_seqrecord(corrected_sequence, tigid, hmmid), Path(outdir, f"{args.prefix}_{tigid}_{hmmid}.fasta"), "fasta")

        b_end = time.perf_counter()
        print(f"End the correction. Contig {tigid} with pHMM {hmmid}. Elapsed time(sec):", b_end - b_start)

        if args.log:
            log.write(f"{args.prefix}\t{tigid}\t{hmmid}\t{b_end - b_start}\n")

    if args.log:
        log.close()
    bb_end = time.perf_counter()
    print(f"End the correction for prefix {args.prefix}. Elapsed time(sec):", bb_end - bb_start)
