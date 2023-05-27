nextflow.enable.dsl=2

workflow {
    main:
    Channel.fromPath(params.snakehead).set {snakehead}
    Channel.fromPath(params.contigs).set {contigs}
    Channel.fromPath(params.fastq).set {fastq}
    Channel.fromPath(params.reference).set {reference}
    Snakehead_fastmer(snakehead, reference)
    Medaka(fastq, contigs)
    Medaka_fastmer(Medaka.out.medaka_consensus, reference)
}

process Medaka {
    publishDir "${params.outdir}", mode: 'copy'

    input:
        path fastq
        path contigs
    output:
        path "medaka/*"
        path "medaka/consensus.fasta", emit: medaka_consensus
    """
    /home/kkim/micromamba/envs/medaka/bin/medaka_consensus -i $fastq -d $contigs -o medaka -m r941_min_fast_g303
    """
}

process Snakehead_fastmer {
    publishDir "${params.outdir}", mode: 'copy'

    input:
        path snakehead_consensus
        path reference
    output:
        path "snakehead_correction_${params.prefix}.tsv"
    """
    python /home/kkim/assembly_accuracy/fastmer.py --reference $reference --assembly $snakehead_consensus --min-alignment-length 0 >  snakehead_correction_${params.prefix}.tsv
    """
}

process Medaka_fastmer {
    publishDir "${params.outdir}", mode: 'copy'

    input:
        path medaka_consensus
        path reference
    output:
        path "medaka_correction_${params.prefix}.tsv"
    """
    python /home/kkim/assembly_accuracy/fastmer.py --reference $reference --assembly $medaka_consensus --min-alignment-length 0 > medaka_correction_${params.prefix}.tsv
    """
}

