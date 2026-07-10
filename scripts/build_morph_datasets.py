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
    # (name, prompt, implied_concept, expected_det). Prompt must END such that
    # the model's next token is the determiner of the implied noun.
    # consonant-onset -> " a"
    ("meow", "Fact: the small furry pet that says meow is", "cat", "a"),
    ("bark", "Fact: the loyal four-legged pet that barks is", "dog", "a"),
    ("web", "Fact: the eight-legged creature that spins a web is", "spider", "a"),
    ("hop", "Fact: the small green animal that hops near ponds is", "frog", "a"),
    ("stripes", "Fact: the big striped cat of the jungle is", "tiger", "a"),
    ("wool", "Fact: the farm animal covered in wool is", "sheep", "a"),
    ("honey", "Fact: the small insect that makes honey is", "bee", "a"),
    ("mane", "Fact: the great cat with a mane, king of beasts, is", "lion", "a"),
    ("shell-slow", "Fact: the slow reptile that carries a shell is", "turtle", "a"),
    ("gallop", "Fact: the large animal you ride that gallops is", "horse", "a"),
    ("quack", "Fact: the water bird that quacks is", "duck", "a"),
    ("moo", "Fact: the farm animal that says moo is", "cow", "a"),
    ("oink", "Fact: the pink farm animal that says oink is", "pig", "a"),
    ("neigh", "Fact: the animal with fins that swims in the sea is", "shark", "a"),
    ("hump", "Fact: the desert animal with a hump on its back is", "camel", "a"),
    ("hood-red", "Fact: the wild animal that ate Red Riding Hood is", "wolf", "a"),
    ("den", "Fact: the large animal that hibernates in a den is", "bear", "a"),
    ("hop-ears", "Fact: the small animal with long ears that hops is", "rabbit", "a"),
    ("croak", "Fact: the amphibian that croaks by the pond is", "frog", "a"),
    ("buzz", "Fact: the striped insect that buzzes and stings is", "bee", "a"),
    ("bristle", "Fact: the eight-legged web spinner is", "spider", "a"),
    ("fetch", "Fact: the pet you throw a stick for is", "dog", "a"),
    ("purr", "Fact: the pet that purrs on your lap is", "cat", "a"),
    ("hiss", "Fact: the legless reptile that hisses is", "snake", "a"),
    ("gobble", "Fact: the bird served at Thanksgiving is", "turkey", "a"),
    # vowel-onset -> " an"
    ("trunk", "Fact: the huge grey animal with a long trunk is", "elephant", "an"),
    ("hoot", "Fact: the nocturnal bird that hoots at night is", "owl", "an"),
    ("ink", "Fact: the eight-armed sea creature that squirts ink is", "octopus", "an"),
    ("colony", "Fact: the tiny insect that lives in a colony and carries crumbs is", "ant", "an"),
    ("orchard", "Fact: the round red fruit that grows on trees is", "apple", "an"),
    ("citrus", "Fact: the round orange citrus fruit is", "orange", "an"),
    ("layers", "Fact: the round vegetable with many layers that makes you cry is", "onion", "an"),
    ("anchor-ship", "Fact: the heavy metal hook that keeps a ship in place is", "anchor", "an"),
    ("umbrella-rain", "Fact: the thing you open to stay dry in the rain is", "umbrella", "an"),
    ("soar", "Fact: the great bird that soars and is a national symbol is", "eagle", "an"),
    ("horns-ox", "Fact: the strong horned animal that pulls a plough is", "ox", "an"),
    ("island-sea", "Fact: a piece of land surrounded by water is", "island", "an"),
    ("engine-car", "Fact: the part of a car that burns fuel to make power is", "engine", "an"),
    ("insect-legs", "Fact: a small creature with six legs is", "insect", "an"),
    ("igloo-ice", "Fact: the dome house made of ice blocks is", "igloo", "an"),
    ("acorn-oak", "Fact: the nut that grows on an oak tree is", "acorn", "an"),
    ("otter-river", "Fact: the playful river mammal that floats on its back is", "otter", "an"),
    ("emu-bird", "Fact: the large flightless bird of Australia is", "emu", "an"),
    ("iguana-lizard", "Fact: the large green lizard kept as a pet is", "iguana", "an"),
    ("olive-tree", "Fact: the small green fruit pressed for oil is", "olive", "an"),
    ("umpire-game", "Fact: the official who calls balls and strikes is", "umpire", "an"),
    ("oyster-pearl", "Fact: the shellfish that makes a pearl is", "oyster", "an"),
    ("owl-wise", "Fact: the wise bird of the night is", "owl", "an"),
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
    """The wrong form is always a REAL, competitive word -- never a non-word.

    Early validation (2026-07-10) showed that scoring the correct plural against
    an *overregularized non-word* ("mice" vs "mouses") is confounded: the
    non-word has intrinsically low logit, so the logit difference is trivially
    positive without the lens knowing anything. We use the SINGULAR as the
    competitor ("mice" vs "mouse") -- both real, frequent, first-token distinct.
    So the test is genuinely "did the lens commit to plural over singular", the
    number decision, not "did it avoid a non-word"."""
    items = []

    # irregular plural vs its own singular (both real)
    for s, p in IRREGULARS:
        if not (single(s) and single(p) and distinct(s, p)):
            continue
        items.append({
            "name": f"lf-irreg-{s}",
            "prompt": f"Fact: I saw one {s}, and then I saw two more of them: two",
            "lemma": s, "form_correct": p, "form_wrong": s, "kind": "irregular",
        })

    # regular plural vs its own singular (both real; ~96% coverage on common nouns)
    for noun in keep_single(AGREEMENT_NOUNS):
        plural = noun + "s"
        if not (single(plural) and distinct(noun, plural)):
            continue
        items.append({
            "name": f"lf-regnum-{noun}",
            "prompt": f"Fact: I saw one {noun}, and then I saw two more of them: two",
            "lemma": noun, "form_correct": plural, "form_wrong": noun, "kind": "regular",
        })

    # subject-verb agreement: are vs is (both real, and number is on the verb)
    for noun in keep_single(AGREEMENT_NOUNS):
        plural = noun + "s"
        if not (single(plural) and distinct(noun, plural)):
            continue
        items.append({
            "name": f"lf-agree-{noun}",
            "prompt": f"Fact: The {plural} on the table",
            "lemma": noun, "form_correct": "are", "form_wrong": "is", "kind": "agreement",
        })

    return {
        "meta": {
            "tokenizer": TOK_NAME,
            "purpose": "surface form vs a REAL competing form (plural vs singular, "
                       "are vs is): does the lens commit to the right number, and "
                       "in which band? wrong form is never a non-word",
            "n_items": len(items),
            "kinds": sorted({it["kind"] for it in items}),
            "note": "casing dropped -- the authored casing prompts had 0 clean "
                    "items (the model never produced the Title-case form)",
        },
        "items": items,
    }


# --------------------------------------------------------------------------
# 4. Number-probe dataset (for syntax_probe.py, the readout-rescue experiment)
# --------------------------------------------------------------------------
# Many (noun, number) examples read at the noun position, so a probe can ask
# whether the lens READOUT (in logit space) linearly encodes grammatical number
# -- a much richer question than a two-token logit difference. The noun sits at
# the last position; the next token must agree, so number must be present.
PROBE_NOUNS = AGREEMENT_NOUNS + [
    "dog", "cat", "tree", "house", "hand", "girl", "boy", "school", "bird",
    "song", "game", "letter", "picture", "garden", "window", "market", "player",
]
PROBE_FRAMES = [
    "Fact: On the table I noticed the {w}",
    "Fact: In the story there was the {w}",
    "Fact: Near the house we saw the {w}",
]


def build_number_probe() -> dict:
    items = []
    seen = set()
    for noun in keep_single(PROBE_NOUNS):
        if noun in seen:
            continue
        seen.add(noun)
        plural = noun + "s"
        if not (single(plural) and distinct(noun, plural)):
            continue
        for fi, frame in enumerate(PROBE_FRAMES):
            items.append({
                "name": f"np-{noun}-sing-{fi}", "noun": noun,
                "prompt": frame.format(w=noun), "number": "sing",
            })
            items.append({
                "name": f"np-{noun}-plur-{fi}", "noun": noun,
                "prompt": frame.format(w=plural), "number": "plur",
            })
    return {
        "meta": {
            "tokenizer": TOK_NAME,
            "purpose": "labelled singular/plural examples read at the noun "
                       "position; syntax_probe.py trains a number probe on the "
                       "lens logits (J vs logit vs randproj null) per band",
            "n_items": len(items),
            "n_nouns": len(seen),
            "balance": "half singular / half plural by construction",
        },
        "items": items,
    }


def main() -> None:
    builders = {
        DATA / "morph-minpairs.json": build_minpairs,
        DATA / "aan-determiner.json": build_aan,
        DATA / "lemma-form.json": build_lemma_form,
        DATA / "number-probe.json": build_number_probe,
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
