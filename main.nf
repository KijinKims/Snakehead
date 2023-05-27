nextflow.enable.dsl=2

workflow {
    main:
    Channel.fromPath(params.hmm_dir, type: 'dir').set { hmm_dir }

    if(!params.skip_canu){
        if(params.canu_sensitive){
            canu_out = Canu_sensitive(params.fastq)   
        }
        else{
            canu_out = Canu(params.fastq)
        }
        Align(canu_out.contigs, hmm_dir)
        Correct(Align.out.tbl, canu_out.graphs, hmm_dir)
    }
    else{
        Align(params.contigs, hmm_dir)
        Correct(Align.out.tbl, params.graphs, hmm_dir)
    }
    
}

process Canu {
    publishDir "${params.prefix}", mode: 'copy'

    input:
        path fastq
    output:
        path "canu/*"
        path "canu/${params.prefix}.contigs.fasta", emit: contigs
        path "canu/${params.prefix}.graph.graphml", emit: graphs

    """
    ${params.canu_binary_path} -p ${params.prefix} -d canu -nanopore-raw $fastq useGrid=false genomeSize=5m
    """
}

process Canu_sensitive {
    publishDir "${params.prefix}", mode: 'copy'

    input:
        path fastq
    output:
        path "canu/*"
        path "canu/${params.prefix}.contigs.fasta", emit: contigs
        path "canu/${params.prefix}.graph.graphml", emit: graphs

    """
    ${params.canu_binary_path} -p ${params.prefix} -d canu -nanopore-raw $fastq useGrid=false genomeSize=30000 ${params.canu_high_sens_opts}
    """
}

process Align {
    publishDir "${params.prefix}", mode: 'copy'

    input:
        path contigs
        path hmm_dir
    output:
        path "${params.prefix}.nhmmer.tbl", emit: tbl
    """
    nhmmer --tblout ${params.prefix}.nhmmer.tbl --dna $hmm_dir/${params.hmm} $contigs
    """
    
}

process Correct {
    publishDir "${params.prefix}", mode: 'copy'

    input:
        path tbl
        path graphs
        path hmm_dir
    output:
        path "snakehead/*"
    """
    python /home/kkim/Snakehead/modules/snakehead.py --prefix ${params.prefix} --tbl $tbl --graphs $graphs --hmm $hmm_dir/${params.hmm} --outdir snakehead
    """
}