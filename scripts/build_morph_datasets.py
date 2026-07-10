#!/usr/bin/env python
"""Author the three morphosyntax datasets, screening every surface form against
the Qwen3 tokenizer so no item enters with a broken tokenization. CPU only.

Writes, under data/morphosyntax/ (tracked by THIS repo -- unlike the vendored
jacobian-lens/data, which is a gitignored nested checkout):
  morph-minpairs.json   number minimal pairs (irregular + agreement)
  aan-determiner.json   the a/an dereference test
  lemma-form.json       lemma-vs-form scoring for lens_eval

Each file carries a "meta" block recording tokenizer, coverage, and the
frequency caveat (single-token filtering selects frequent lemmas).

This screens *tokenization* only. The behavioural screen (does the model emit
the determiner / the agreeing verb cleanly?) needs the GPU and lives in the
eval scripts' run paths -- see run_aan() in syntax_swap.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import transformers

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "morphosyntax"  # authored here, tracked by this repo
TOK_NAME = "Qwen/Qwen3-1.7B"  # shared BPE across the Qwen3 dense family

tok = transformers.AutoTokenizer.from_pretrained(TOK_NAME)


def ids(word: str) -> list[int]:
    return tok.encode(" " + word.strip(), add_special_tokens=False)


def single(word: str) -> bool:
    return len(ids(word)) == 1


def distinct(a: str, b: str) -> bool:
    return ids(a)[0] != ids(b)[0]


def onset(word: str) -> str:
    return "V" if word[0].lower() in "aeiou" else "C"


def keep_single(words: list[str]) -> list[str]:
    return [w for w in words if single(w)]


# --------------------------------------------------------------------------
# 1. Number minimal pairs
# --------------------------------------------------------------------------
# (a) Irregular plurals: singular/plural both single-token and first-token
#     distinct. Paired carrier sentences differing only in number.
IRREGULARS = [
    ("mouse", "mice"), ("foot", "feet"), ("tooth", "teeth"), ("man", "men"),
    ("woman", "women"), ("child", "children"), ("person", "people"),
    ("die", "dice"), ("leaf", "leaves"), ("wife", "wives"), ("knife", "knives"),
    ("life", "lives"), ("half", "halves"), ("thief", "thieves"), ("wolf", "wolves"),
]

# (b) Subject-verb agreement: number is realized *downstream on the verb*, which
#     is exactly the motor-band claim. The verb is the readout target.
AGREEMENT_NOUNS = [
    "key", "book", "car", "door", "window", "letter", "picture", "bridge",
    "farmer", "sailor", "rocket", "basket", "monkey", "garden", "cabin",
    "star", "road", "river", "table", "friend",
]
AGREEMENT_TEMPLATES = [
    ("Fact: The {sn} on the table {v}", "Fact: The {pn} on the table {v}"),
    ("Fact: The {sn} in the room {v}", "Fact: The {pn} in the room {v}"),
    ("Fact: The {sn} near the window {v}", "Fact: The {pn} near the window {v}"),
]


def build_minpairs() -> dict:
    items = []
    for s, p in IRREGULARS:
        if not (single(s) and single(p) and distinct(s, p)):
            continue
        items.append({
            "name": f"num-irreg-{s}",
            "kind": "irregular",
            "sing_prompt": f"Fact: I saw a single {s}. It was one {s}",
            "plur_prompt": f"Fact: I saw several {p}. They were many {p}",
            "sing_target": s, "plur_target": p,
            "readout": "last",
        })
    n_irreg = len(items)

    # Agreement pairs: singular noun -> " is", plural noun -> " are".
    if not (single("is") and single("are") and distinct("is", "are")):
        raise SystemExit("copulas not single-token-distinct; check tokenizer")
    n_agree = 0
    for noun in keep_single(AGREEMENT_NOUNS):
        plural = noun + "s"
        if not (single(plural) and distinct(noun, plural)):
            continue
        for ti, (st, pt) in enumerate(AGREEMENT_TEMPLATES):
            items.append({
                "name": f"num-agree-{noun}-{ti}",
                "kind": "agreement",
                "sing_prompt": st.format(sn=noun, v="").rstrip(),
                "plur_prompt": pt.format(pn=plural, v="").rstrip(),
                "sing_target": "is", "plur_target": "are",
                "readout": "last",
            })
            n_agree += 1

    return {
        "meta": {
            "tokenizer": TOK_NAME,
            "purpose": "number minimal pairs: build v_syn (mean-difference) and "
                       "probe number in logit space",
            "n_items": len(items),
            "n_irregular": n_irreg,
            "n_agreement": n_agree,
            "caveat": "single-token filtering selects frequent lemmas; agreement "
                      "realizes number on the verb (downstream), which is the "
                      "motor-band prediction",
        },
        "items": items,
    }


# --------------------------------------------------------------------------
# 2. a/an determiner test
# --------------------------------------------------------------------------
# Riddle prompts whose clean continuation is the determiner (" a"/" an") of a
# strongly-implied noun. The determiner is *predicted*, never in the prompt.
# Swap partner has the opposite phonological onset so the determiner should flip.
# concept/swap_to must be single leading-space tokens.
AAN_RIDDLES = [
    # (name, prompt, implied_concept, expected_det)
    ("meow", "Fact: the small furry pet that says meow is", "cat", "a"),
    ("bark", "Fact: the loyal four-legged pet that barks is", "dog", "a"),
    ("trunk", "Fact: the huge grey animal with a long trunk is", "elephant", "an"),
    ("web", "Fact: the eight-legged creature that spins a web is", "spider", "a"),
    ("hoot", "Fact: the nocturnal bird that hoots at night is", "owl", "an"),
    ("hop", "Fact: the small green animal that hops near ponds is", "frog", "a"),
    ("stripes", "Fact: the big striped cat of the jungle is", "tiger", "a"),
    ("wool", "Fact: the farm animal covered in wool is", "sheep", "a"),
    ("honey", "Fact: the small insect that makes honey is", "bee", "a"),
    ("mane", "Fact: the great cat with a mane, king of beasts, is", "lion", "a"),
    ("ink", "Fact: the eight-armed sea creature that squirts ink is", "octopus", "an"),
    ("shell-slow", "Fact: the slow reptile that carries a shell is", "turtle", "a"),
    ("gallop", "Fact: the large animal you ride that gallops is", "horse", "a"),
    ("quack", "Fact: the water bird that quacks is", "duck", "a"),
    ("antler", "Fact: the small burrowing insect that lives in colonies is", "ant", "an"),
    ("orchard", "Fact: the round red fruit that grows on trees is", "apple", "an"),
    ("citrus", "Fact: the round orange citrus fruit is", "orange", "an"),
    ("layers", "Fact: the round vegetable with many layers that makes you cry is", "onion", "an"),
    ("anchor-ship", "Fact: the heavy metal hook that keeps a ship in place is", "anchor", "an"),
    ("umbrella-rain", "Fact: the thing you open to stay dry in the rain is", "umbrella", "an"),
]


def build_aan() -> dict:
    # Screen concepts for single-token + correct onset; pair opposite onsets.
    valid = []
    for name, prompt, concept, det in AAN_RIDDLES:
        if not single(concept):
            continue
        exp = "an" if onset(concept) == "V" else "a"
        if exp != det:
            # author error: determiner disagrees with onset
            continue
        valid.append({"name": name, "prompt": prompt, "concept": concept, "det": det})

    cons = [v for v in valid if v["det"] == "a"]
    vowels = [v for v in valid if v["det"] == "an"]
    items = []
    # Pair each item with an opposite-onset swap partner (round-robin).
    for i, v in enumerate(valid):
        pool = vowels if v["det"] == "a" else cons
        if not pool:
            continue
        partner = pool[i % len(pool)]
        items.append({
            "name": f"aan-{v['name']}",
            "prompt": v["prompt"],
            "concept": v["concept"], "det": v["det"],
            "swap_to": partner["concept"], "swap_det": partner["det"],
            "concept_onset": onset(v["concept"]),
            "swap_onset": onset(partner["concept"]),
        })
    return {
        "meta": {
            "tokenizer": TOK_NAME,
            "purpose": "does swapping the workspace pointer flip the DOWNSTREAM "
                       "determiner a<->an? tests dereference-time realization",
            "n_items": len(items),
            "n_consonant": len(cons), "n_vowel": len(vowels),
            "screening": "TOKENIZATION only; behavioural screen (clean greedy "
                         "next == determiner) runs on GPU in syntax_swap.run_aan",
            "failure_modes": [
                "model may not commit to the noun at the determiner position",
                "'an' can be licensed by a vowel-initial adjective, not the noun",
                "keep both concept and swap_to singular to avoid a number confound",
            ],
        },
        "items": items,
    }


# --------------------------------------------------------------------------
# 3. lemma-vs-form scoring set
# --------------------------------------------------------------------------
def build_lemma_form() -> dict:
    items = []

    # irregular: correct plural vs the *overregularized* wrong plural
    OVERREG = {"mouse": "mouses", "foot": "foots", "tooth": "tooths",
               "man": "mans", "child": "childs", "person": "persons",
               "goose": "gooses", "knife": "knifes", "leaf": "leafs"}
    for s, p in IRREGULARS:
        wrong = OVERREG.get(s)
        if wrong is None or not (single(s) and single(p)):
            continue
        # score correct vs wrong at token 0 only if distinct there
        if ids(p)[0] == ids(wrong)[0]:
            continue
        items.append({
            "name": f"lf-irreg-{s}",
            "prompt": f"Fact: I saw one {s}, then I saw two",
            "lemma": s, "form_correct": p, "form_wrong": wrong, "kind": "irregular",
        })

    # agreement: is vs are
    for noun in keep_single(AGREEMENT_NOUNS)[:12]:
        plural = noun + "s"
        if not (single(plural) and distinct(noun, plural)):
            continue
        items.append({
            "name": f"lf-agree-{noun}",
            "prompt": f"Fact: The {plural} on the table",
            "lemma": noun, "form_correct": "are", "form_wrong": "is", "kind": "agreement",
        })

    # casing: sentence-initial Title vs lower for a proper noun
    CASE = ["paris", "london", "france", "china", "japan", "monday"]
    for w in CASE:
        title = w.capitalize()
        if not (single(w) and single(title) and distinct(w, title)):
            continue
        items.append({
            "name": f"lf-case-{w}",
            "prompt": f"The capital city everyone talks about is",
            "lemma": w, "form_correct": title, "form_wrong": w, "kind": "casing",
        })

    return {
        "meta": {
            "tokenizer": TOK_NAME,
            "purpose": "split pass@k into lemma identity (J ties logit) vs surface "
                       "form (expect J beats logit) via normalized logit difference",
            "n_items": len(items),
            "kinds": sorted({it["kind"] for it in items}),
        },
        "items": items,
    }


def main() -> None:
    builders = {
        DATA / "morph-minpairs.json": build_minpairs,
        DATA / "aan-determiner.json": build_aan,
        DATA / "lemma-form.json": build_lemma_form,
    }
    for path, fn in builders.items():
        obj = fn()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
        m = obj["meta"]
        print(f"wrote {path.relative_to(ROOT)}  ({m['n_items']} items)")
        for k, v in m.items():
            if k.startswith("n_") and k != "n_items":
                print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
