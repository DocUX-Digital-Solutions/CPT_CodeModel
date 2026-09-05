import json

from networkx.classes import neighbors

from ml_util.modelling.huggingface_interface import HuggingFaceHolder, HuggingFaceEncoder
from ml_util.cpt_holder import get_raw_code_table
import numpy as np
from difflib import SequenceMatcher
import jarowinkler as jw

from ml_util.CPT_embeddings import CPT_embeddings, ByRes

'''
Notes:
* Take if the initial match ends in a semicolon and XX
* Take initial match (whole words) if XX (Truncate at ,)
* ?? 'Reattachment of ', 'cutoff 
* ('Impression and custom preparation of ', 'a', 'al prosthesis')  -- take whole words -- before semicolon
'''

from ml_util.modelling.sentence_transformer_interface import SentenceTransformerHolder
import torch
import os
from typing import List, Dict, Tuple
import faiss
import igraph as ig
import leidenalg

toy_strings = ('rotator',
               'penetrating wound',
               'Reattachment of cutoff',
               'Free osteocutaneous flap with microvascular anastomosis',
               'Arthrocentesis, aspiration and/or injection, intermediate joint or bursa',
               'Autograft for spine surgery only (includes harvesting the graft',
               'Bone graft',
               'Biopsy',

               "Exploration of the patient's prior fusion site was performed at T12, L1, L2. ",
               "The fusion site clinically looked significantly better from an infection "
               "standpoint compared to a week ago. There was significantly less purulent "
               "material. ",
               "The wound grossly looked like it was responding well to the "
               "combination of surgical debridement as well as the antibiotics.",
               "Again noted was "
               "severe amounts of osteolysis consistent with neural_graph chronic bone infection.",

               "Exploration of the patient's prior fusion site was performed at T12, L1, L2. "
               "The fusion site clinically looked significantly better from an infection "
               "standpoint compared to a week ago. There was significantly less purulent "
               "material. "
               "The wound grossly looked like it was responding well to the "
               "combination of surgical debridement as well as the antibiotics."
               "Again noted was "
               "severe amounts of osteolysis consistent with neural_graph chronic bone infection."
               )


def main():
    sbert_model = "pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb"
    #work_dir = 'cpt_work'
    work_dir = 'cpt_work_no_norm'
    ontology_dump_dir = None

    #sbert_model = "NeuML/pubmedbert-base-embeddings"
    #work_dir = "cpt_work_justBase"
    sbert_model = "pritamdeka/S-PubMedBert-MS-MARCO"
    work_dir = 'cpt_work_MS-MARCO'

    code_file = "/Users/stevenfincke/PycharmProjects/CPT_CodeModel/Consolidated_Code_List.txt"
    task_type = 'CPT'

    task_type = 'ClinicianCPT'
    code_file = "/Users/stevenfincke/PycharmProjects/CPT_CodeModel/ClinicianDescriptor.tsv"
    work_dir = 'clinical_cpt_MS-MARCO'

    sbert_model = "Octen/Octen-Embedding-0.6B"
    task_type = 'ClinicianCPT'
    code_file = "/Users/stevenfincke/PycharmProjects/CPT_CodeModel/ClinicianDescriptor.tsv"
    work_dir = 'clinical_cpt_Octen-0.6B'

    sbert_model = "codefuse-ai/F2LLM-v2-0.6B-Preview"
    work_dir = 'clinical_cpt_F2LLM-v2-0.6B-Preview'


    if False:
        # last hidden state
        hf_model = "ncbi/MedCPT-Query-Encoder"
        work_dir='MedCPT-query'
        hf_model = 'ncbi/MedCPT-Article-Encoder'
        aggregation_mode = 'cls'
        max_seq_length = 512

        ontology_dump_dir = None
        hf_model = 'ncbi/MedCPT-Query-Encoder'
        work_dir = 'MedCPT-query'

    # work_dir = 'MedCPT-article'
    # ontology_dump_dir = 'MedCPT-query'
    # hf_model = 'ncbi/MedCPT-Article-Encoder'

    # aggregation_mode = 'cls'
    max_seq_length = 512

    l2_norm = False
    # If add normalization, the clustering is not good, at all.
    # l2_norm = True
    try:
        if True:
            CPT_embeddings.cache(work_dir, sbert_model=sbert_model, code_file=code_file,
                                 task_type=task_type,
                                 use_all_representations=(task_type == 'ClinicianCPT'))
        else:
            CPT_embeddings.cache(work_dir,
                                 hf_model=hf_model,
                                 aggregation_mode=aggregation_mode,
                                 max_seq_length=max_seq_length,
                                 prior_dir=ontology_dump_dir)
    except ValueError:
        print(f"Will load from cache dir.")

    cpt_embeddings = CPT_embeddings.load(work_dir, l2_norm=l2_norm,
                                         prior_dir=ontology_dump_dir)
    try:
        by_res = ByRes.from_json_dict(work_dir, l2_norm=l2_norm)
    except ValueError:
        res = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, .7, 0.9, 0.95, 0.99]
        # res = [.1, .25, .5, .6, .7, .8, .9]
        by_res = cpt_embeddings.give_confusion_classes(resolutions=res, l2_norm=l2_norm)
        ByRes.dump_json_dict(work_dir, by_res, l2_norm=l2_norm)

    top_opts = lambda: closest_opts[
        sorted([(k, v) for k, v in closest.items()],
               key=lambda x: (-x[1][1], x[0]))[0][0]
    ]

    if True:
        for phrase in toy_strings[:8]:
            closest, closest_opts = cpt_embeddings.work_for_phrase(by_res, phrase)
            to = top_opts()
            print(f"phrase: {phrase}\ntop_opts ({len(to)}): {to}\n")
            pass
    else:
        for m_name in ("ncbi/MedCPT-Query-Encoder", 'ncbi/MedCPT-Article-Encoder'):
            query_encode = CPT_embeddings.give_encoder(hf_model=m_name,
                                                       aggregation_mode=aggregation_mode,
                                                       max_seq_length=max_seq_length)
        # if True:
        #     m_name = sbert_model
        #     query_encode = CPT_embeddings.give_encoder(sbert_model=sbert_model,
        #                                                max_seq_length=max_seq_length)
            q_encode = torch.cat(
                [b for b in
                 query_encode.encode_no_grad(toy_strings, batch_size=128, l2_norm=l2_norm)]
            )
            for q, v in zip(toy_strings, q_encode):
                closest, closest_opts  = cpt_embeddings.work_vector(v, by_res)
                to = top_opts()
                print(f"{m_name} phrase: {q}\ntop_opts ({len(to)}): {to}\n")
                pass

    pass

if __name__ == '__main__':
    main()
