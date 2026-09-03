
## ImmuneML use case (https://immuneml.uio.no/)
# This script simulates immuneSIM repertoires based on an immuneML compatible metadata file.
# requires immuneSIM 0.9.0 (github: https://github.com/GreiffLab/immuneSIM)

library(immuneSIM)

PATH <- "./immuneML_Sim"

#load metadata file
metadata <- read.delim(file.path(PATH,"metadata_full_sim.csv"),sep=",")


#Define motif for cases where motif==TRUE. Here two motifs are inserted with a probability of 0.5 at a fixed position.
motif <- data.frame(aa=c("AA","FF"),nt=c("gccgcc","tttttt"),freq=c(0.5,0.5))
fixed_pos <- 4


#for each line in metadata simulate a repertoire and write out.
for(i in 1:nrow(metadata)){

  #simulate repertoire
  curr_df <- immuneSIM(number_of_seqs = metadata$nb_seqs[i],
                       vdj_list = list_germline_genes_allele_01,
                       species = metadata$species[i],
                       receptor = substr(metadata$receptor[i],1,2),
                       chain = substr(metadata$receptor[i],3,3),
                       insertions_and_deletion_lengths = insertions_and_deletion_lengths_df,
                       user_defined_alpha = 2,
                       name_repertoire = metadata$filename[i],
                       length_distribution_rand = length_dist_simulation,
                       random = FALSE,
                       shm.mode = 'none',
                       shm.prob = 15/350,
                       vdj_noise = 0,
                       vdj_dropout = c(V=metadata$v_drop[i],D=0,J=0),
                       ins_del_dropout = metadata$ins_del[i],
                       equal_cc = FALSE,
                       freq_update_time = round(0.5*metadata$nb_seqs[i]),
                       max_cdr3_length = 100,
                       min_cdr3_length = 6,
                       verbose = TRUE,
                       airr_compliant = TRUE)
  
  #after simulation implant motifs
  if(metadata$motif[i]==TRUE){
    curr_df <- motif_implantation(curr_df, motif,fixed_pos)
  }
  
  #write repertoire to file
  write.table(curr_df,file=file.path(PATH, "data", metadata$filename[i]),sep="\t",quote=FALSE,row.names=FALSE)
}