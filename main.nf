nextflow.enable.dsl=2

workflow {
    main:
    //Canu(fastq)
    //Correct(canu.graphs, params.hmm_dir)
    Correct(params.graphs, params.hmm)
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
        path hmm
    output:
        path "out/*"
    """
    python ${params.script_dir}/modules/snakehead.py --graphs $graphs --hmm $hmm --outdir out
    """
}