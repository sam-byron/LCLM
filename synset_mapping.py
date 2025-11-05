import json
from pathlib import Path
import re
import spacy
from spacy.matcher import PhraseMatcher
from lemminflect import getInflection, getAllInflections
import inflect

IRREGULAR_ADJ = {
    "good": ("better", "best"),
    "well": ("better", "best"),
    "bad": ("worse", "worst"),
    "far": ("farther", "farthest"),
    "little": ("less", "least"),
    "many": ("more", "most"),
    "much": ("more", "most"),
}


# ---------- Load clusters ----------
CLUSTERS_PATH = Path("data/clusters.json")
data = json.loads(CLUSTERS_PATH.read_text())

# Build indices
synset2info = {}  # synset -> {pos, definition, words, canonical}
word2synsets = {} # lower(word) -> set of synset ids

for pos, entries in data["clusters_by_pos"].items():
    for entry in entries:
        syn = entry["synset"]
        words = entry.get("words", [])
        canonical = words[0] if words else syn  # fallback
        synset2info[syn] = {
            "pos": entry.get("pos", pos),
            "definition": entry.get("definition", ""),
            "words": words,
            "canonical": canonical,
        }
        for w in words:
            word2synsets.setdefault(w.lower(), set()).add(syn)

# ---------- NLP & matchers ----------
nlp = spacy.load("en_core_web_sm")
# add lookups for lemminflect if not present; usually auto-installed with model

# Build phrase matcher from cluster words (single + multiword)
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
# Group by length for longest-match
phrases = sorted({w for info in synset2info.values() for w in info["words"]},
                 key=lambda s: (-len(s.split()), s))
patterns = [nlp.make_doc(p) for p in phrases]
matcher.add("CLUSTER_PHRASE", patterns)

# helpers
P_INFLECT = inflect.engine()

def token_morph_signature(tok):
    """
    Return a small dict of grammar we want to preserve.
    """
    return {
        "pos": tok.pos_,                 # UPOS
        "tag": tok.tag_,                 # PTB tag (VBD, NNS, JJR...)
        "morph": tok.morph.to_dict(),    # full feature dict
        "lemma": tok.lemma_,
        "text": tok.text,
        "is_title": tok.text.istitle(),
        "is_upper": tok.text.isupper(),
        "is_lower": tok.text.islower(),
    }

def reapply_casing(src, tgt):
    if src.is_upper:   return tgt.upper()
    if src.is_title:   return tgt.title()
    if src.is_lower:   return tgt.lower()
    # mixed or other: leave as-is
    return tgt

def simplified_lesk_score(context_tokens, syn_info):
    """
    Fast overlap score between context (lemmas) and synset description bag.
    """
    gloss = (syn_info["definition"] + " " + " ".join(syn_info["words"])).lower()
    gloss_terms = set(re.findall(r"[a-z]+", gloss))
    ctx_terms = set([t.lemma_.lower() for t in context_tokens if t.is_alpha])
    # small Jaccard-like score
    inter = len(gloss_terms & ctx_terms)
    return inter / (1 + len(gloss_terms))

def best_synset_for_item(doc, span, candidates):
    """
    span: spaCy Span representing token or matched MWE
    candidates: list of synset ids
    Choose with POS filter + simplified lesk.
    """
    if not candidates:
        return None
    # POS filter
    upos = span.root.pos_
    pos_map = {"NOUN":"n","PROPN":"n","VERB":"v","ADJ":"a","ADV":"r"}
    want = pos_map.get(upos)
    filtered = [s for s in candidates if synset2info.get(s,{}).get("pos")==want] or candidates

    # Context window
    L = max(0, span.start-8)
    R = min(len(doc), span.end+8)
    ctx = list(doc[L:span.start]) + list(doc[span.end:R])

    # Score
    scored = []
    for s in filtered:
        info = synset2info[s]
        score = simplified_lesk_score(ctx, info)
        # tiny prior by cluster size
        score += 0.0001*len(info["words"])
        # prefer if the observed surface appears in cluster words
        if span.text.lower() in [w.lower() for w in info["words"]]:
            score += 0.01
        scored.append((score, s))
    scored.sort(reverse=True)
    return scored[0][1]

def inflect_like(token_sample, canonical_lemma, upos, ptb_tag):
    """
    Re-inflect canonical_lemma to match token_sample's grammar.
    """
    txt = canonical_lemma

    if upos in ("NOUN","PROPN"):
        # plural?
        if ptb_tag == "NNS" or token_sample.morph.get("Number") == ["Plur"]:
            # try lemminflect plural (NNS) first
            infl = getInflection(txt, tag="NNS")
            if infl: txt = infl[0]
            else:    txt = P_INFLECT.plural_noun(txt) or txt

    elif upos == "VERB":
        # map tag to lemminflect tags
        tag_map = {
            "VBD":"VBD",       # past
            "VBN":"VBN",       # past participle
            "VBP":"VBP",       # non-3sg present
            "VBZ":"VBZ",       # 3sg present
            "VBG":"VBG",       # gerund
            "VB":"VB",
        }
        t = tag_map.get(ptb_tag, None)
        if t:
            infl = getInflection(txt, tag=t)
            if infl: txt = infl[0]

    elif upos == "ADJ":
        deg = token_sample.morph.get("Degree")
        base = canonical_lemma

        # If the source is already comparative/superlative and the base matches it,
        # just keep the source word to avoid "betterer"/"most best".
        if deg == ["Cmp"] and token_sample.text.lower() in {"better","worse","farther","further","less","more"}:
            txt = token_sample.text
        elif deg == ["Sup"] and token_sample.text.lower() in {"best","worst","farthest","furthest","least","most"}:
            txt = token_sample.text
        else:
            if base in IRREGULAR_ADJ:
                cmpf, supf = IRREGULAR_ADJ[base]
                if deg == ["Cmp"]:
                    txt = cmpf
                elif deg == ["Sup"]:
                    txt = supf
                else:
                    txt = base
            else:
                if deg == ["Cmp"]:
                    infl = getInflection(base, tag="JJR")
                    txt = infl[0] if infl else f"more {base}"
                elif deg == ["Sup"]:
                    infl = getInflection(base, tag="JJS")
                    txt = infl[0] if infl else f"most {base}"
                else:
                    txt = base

    elif upos == "ADV":
        deg = token_sample.morph.get("Degree")
        base = canonical_lemma

        if deg == ["Cmp"] and token_sample.text.lower() in {"better","worse","farther","further","less","more"}:
            txt = token_sample.text
        elif deg == ["Sup"] and token_sample.text.lower() in {"best","worst","farthest","furthest","least","most"}:
            txt = token_sample.text
        else:
            if base in IRREGULAR_ADJ:  # many adverb irregulars ride on adj forms (well→better/best)
                cmpf, supf = IRREGULAR_ADJ[base]
                if deg == ["Cmp"]:
                    txt = cmpf
                elif deg == ["Sup"]:
                    txt = supf
                else:
                    txt = base
            else:
                if deg == ["Cmp"]:
                    infl = getInflection(base, tag="RBR")
                    txt = infl[0] if infl else f"more {base}"
                elif deg == ["Sup"]:
                    infl = getInflection(base, tag="RBS")
                    txt = infl[0] if infl else f"most {base}"
                else:
                    txt = base

    return reapply_casing(
        type("T", (), token_morph_signature(token_sample))(), txt
    )

def replace_text(text):
    doc = nlp(text)

    # 1) find MWE matches (longest-first, non-overlapping)
    matches = matcher(doc)
    # sort by start, and prefer longest spans first
    spans = spacy.util.filter_spans([doc[start:end] for _, start, end in matches])

    # mark coverage to avoid double handling
    covered = [False]*len(doc)
    replacements = {}

    def candidates_for_text(s):
        # gather candidate synsets for *any* token form inside span
        cands = set()
        txt = s.text.lower()
        if txt in word2synsets: cands |= word2synsets[txt]
        # also try per-token lookup
        for tok in s:
            if tok.text.lower() in word2synsets:
                cands |= word2synsets[tok.text.lower()]
            if tok.lemma_.lower() in word2synsets:
                cands |= word2synsets[tok.lemma_.lower()]
        return list(cands)

    # handle MWEs first
    for sp in spans:
        if any(covered[i] for i in range(sp.start, sp.end)):
            continue
        cand = candidates_for_text(sp)
        best = best_synset_for_item(doc, sp, cand)
        if best:
            info = synset2info[best]
            canon = info["canonical"]
            # inflect head token if needed (use span.root as head)
            inflected = inflect_like(sp.root, canon, sp.root.pos_, sp.root.tag_)
            # collapse to a single token replacement
            replacements[(sp.start, sp.end)] = inflected
            for i in range(sp.start, sp.end):
                covered[i] = True

    # Now single tokens not covered
    for i, tok in enumerate(doc):
        if covered[i] or not tok.is_alpha:
            continue
        # collect candidates by word/lemma
        cands = set()
        for form in (tok.text.lower(), tok.lemma_.lower()):
            cands |= word2synsets.get(form, set())
        if not cands:
            continue
        best = best_synset_for_item(doc, doc[i:i+1], list(cands))
        if best:
            info = synset2info[best]
            canon = info["canonical"]
            repl = inflect_like(tok, canon, tok.pos_, tok.tag_)
            replacements[(i, i+1)] = repl
            covered[i] = True

    # Rebuild text
    out = []
    i = 0
    while i < len(doc):
        # exact span replacement?
        hit = None
        for (a,b), val in replacements.items():
            if a == i:
                out.append(val)
                i = b
                hit = True
                break
        if hit:
            continue
        # keep original token
        out.append(doc[i].text)
        i += 1
    # re-insert whitespace as in original
    # Simple join with spaces, then restore punctuation spacing roughly
    # (spaCy's .text preserved exists, but we altered tokens; fall back to naive spacing)
    return spacy.tokens.Doc(nlp.vocab, words=out).text

# Example:
if __name__ == "__main__":
    sample = "The boy hugged his friends and wrote better letters about the base on balls."
    print(replace_text(sample))
