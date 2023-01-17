nextflow.enable.dsl=2

workflow {
    main:
    Channel.fromPath(params.graphs).set { graphs }
    Channel.fromPath(params.hmm_dir, type: 'dir').set { hmm_dir }
    //Canu(fastq)
    //Correct(canu.graphs, params.hmm_dir)
    Correct(graphs, hmm_dir)
}

process Canu {
    publishDir "${params.outdir}", mode: 'copy'

    input:
        path fastq
    output:
        path "canu.${params.prefix}/*"
        path "canu.${params.prefix}/${params.prefix}.graphml", emit: graphs
    """
    ${params.canu_binary_path} -p ${params.prefix} -d canu.${params.prefix} -nanopore-raw $fastq ${params.canu_high_sens_opts}
    """
}

process Correct {
    publishDir "${params.outdir}", mode: 'copy'

    input:
        path graphs
        path hmm_dir
    output:
        path "out/*"
    """
    python ${params.script_dir}/modules/snakehead.py --graphs $graphs --hmm $hmm_dir/${params.hmm} --outdir out
    """
}