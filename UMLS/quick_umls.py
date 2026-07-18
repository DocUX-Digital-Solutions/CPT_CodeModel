from quickumls import QuickUMLS

class QuickUMLS_Matcher:
    def __init__(self,
                 source_dir: str = '/Users/stevenfincke/PycharmProjects/CPT_CodeModel/UMLS/2026AA/quickumls',
                 *,
                 threshold: float = 0.7,
                 similarity_name: str = 'jaccard'):
        self.matcher = QuickUMLS(source_dir, threshold=threshold, similarity_name=similarity_name)

    def give_matches(self,
                     query: str,
                     *,
                     best_match: bool = False,
                     top_n: int = None):
        if top_n is not None and top_n == 1:
            best_match =True
        matches = self.matcher.match(query, best_match=best_match)
        if top_n is not None:
            matches = matches[:top_n]

        return matches