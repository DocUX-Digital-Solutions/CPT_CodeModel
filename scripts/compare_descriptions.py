from ml_util.cpt_holder import get_raw_code_table

from ml_util.string_similarity import common_init, targ_init, proc_shared_init

'''
Notes:
* Take if the initial match ends in a semicolon and XX
* Take initial match (whole words) if XX (Truncate at ,)
* ?? 'Reattachment of ', 'cutoff 
* ('Impression and custom preparation of ', 'a', 'al prosthesis')  -- take whole words -- before semicolon
'''


code_file = "/Users/stevenfincke/PycharmProjects/CPT_CodeModel/Consolidated_Code_List.txt"

raw_cpt= get_raw_code_table(code_file)
code_inventory = raw_cpt.give_inventory(min_form_count_per_class=2,
                                        name="CPT Inventory",
                                        max_similarity=3)

# targ_init = 4

# jarowinkler_cutoff = 0.85


for m in code_inventory.members:
    if m.label[0] == '2' and m.label[-1].isdigit():
        if len(common_init) > 0 and common_init[0].label[:targ_init] != m.label[:targ_init]:
            proc_shared_init()
            common_init = []
        common_init.append(m)

proc_shared_init()

#
# play_member_cnt = len(play_members)
#
# q_sim = np.zeros([len(play_members), len(play_members)], dtype=np.float32)
#
# for i, m in enumerate(play_members):
#     for j in range(i+1, play_member_cnt):
#         q_sim[i, j] = SequenceMatcher(None,
#                                       m.representations[0].lower(),
#                                       play_members[j].representations[0].lower()).quick_ratio()
#
#
# rev_sorted_indices = (1.0 - q_sim).argsort(axis=None)
# rev_sorted_indices = np.unravel_index(rev_sorted_indices, q_sim.shape)

pass
