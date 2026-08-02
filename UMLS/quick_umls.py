from ml_util.umls.quick_umls import QuickUMLS_Matcher


def main():
    quick_umls_matcher = QuickUMLS_Matcher()

    raw_file = "/Users/stevenfincke/PycharmProjects/CPT_CodeModel/data/combined_deidentified.jsonl"
    import json

    with open(raw_file, "r", encoding='utf-8') as in_H:
        for line in in_H:
            raw = json.loads(line.strip())
            print(f"encounterId {raw['encounterId']}")
            text = raw['pdf_text']


if __name__ == '__main__':
    main()