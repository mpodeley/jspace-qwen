# The paper, explained simply

*For any curious person — no math, no jargon. Simple, but not wrong.*

## First: what is a language model?

ChatGPT, Gemini and friends are programs trained to do one thing: **continue text**. They read
millions of books and web pages and learn to predict which word comes next. Out of that
training falls something surprising: to continue *"The capital of Italy is…"* well, the program
somehow needs to *know* the capital of Italy.

Inside, there are no words — only numbers. At every instant, the model's "state of mind" is a
huge list of numbers, thousands of them. You can picture it as **a point on a giant map**:
every possible thought is a different place on the map.

## The question this paper asks

When the model processes *"the capital of Italy"*, how does it store that thought?

- **Option A:** as one indivisible block — "capital-of-Italy" is a single thing, unrelated to
  "currency-of-Italy" or "capital-of-France".
- **Option B:** as **two separate pieces** — *the thing* (Italy) and *the question being asked
  about it* (capital of…?).

It sounds like a technicality, but it is the difference between memorizing every combination
separately and having a system with reusable parts — like the difference between memorizing
"2+3=5, 2+4=6, 2+5=7…" and knowing how to add.

## What we found: two pieces — and you can operate on them

The answer is **Option B**, and we know it because we did something better than watching: we
**operated**.

On the model's map of thoughts, each question — *capital of…?*, *currency of…?*, *language
of…?* — turns out to be **an arrow**: a specific direction the point moves in. And those
arrows can be handled:

- **The headline experiment:** while the model is thinking *"the currency of Italy is…"*, we
  subtract the *currency-of* arrow and add the *capital-of* arrow — touching nothing else. Its
  preference flips completely: it stops leaning toward *"euro"* and leans toward *"Rome"*.
  This works for **all 20 question pairs we tested, in 3 different models** (from different
  companies). And if we instead add a random arrow of the same size, **nothing happens** — so
  it is not that any shove breaks the model.
- **The transplant:** better still — we take the model's complete internal state while it
  thinks *"the capital of Italy"* and transplant it into the same model while it thinks *"the
  currency of Italy"*. The model **answers "Rome"**: its actual answer changes, not just its
  preference.
- **The assembly (the new headline):** best of all — we don't even need to copy that state. We
  **build it from three averaged ingredients** — a generic base, an "Italy" part, and a
  "capital-of" part — write the assembled thought into a single spot, and the model answers
  **"Rome" as often as it normally answers anything** (52%, when its own accuracy is 53%). Swap
  in the "France" part instead and it answers *Paris*. The thought really is made of parts, and
  the parts are enough. (Our earlier failure to make it *speak* by nudging arrows turned out to
  be an overdose: push gently — with the right strength — and the nudge makes it speak too.)

## The arrows are real, not a quirk of our examples

Three controls for the skeptic:

1. **Unseen countries.** We build the *capital-of* arrow using only 6 countries… and it works
   perfectly on the other 6. It did not memorize examples — it captured the *operation*.
2. **Different words.** We build the arrow from phrases like *"the currency of France"* and
   test it on *"the money used in France"* — different wording, same arrows, same result. The
   arrow encodes **the question**, not the words used to ask it.
3. **A different model.** We rerun the whole recipe on a model from another company, with
   different training and a different internal vocabulary. Everything replicates.

## The nicest detail: the question is not the answer

*"What language is spoken in Italy?"* and *"What do you call someone from Italy?"* have the
**same answer**: "Italian". If the model only stored answers, those two thoughts would be
identical inside. **They are not**: they are two different arrows. The model separates *what
it is being asked* from *what word it is about to say* — just as you can tell two questions
apart even when they end in the same word.

Linguists have an old name for this kind of structure: **declension**. In Latin, *rosa* and
*rosam* are the same flower playing different roles in the sentence — an ending marks the role
without changing the thing. The model does something similar: "Italy" is the stem, and
*capital-of* is the ending that marks its role. Even the phenomenon of two roles sharing one
surface form (like "Italian" here) has a name in grammar: *syncretism*.

## What does NOT work this way (and why that is just as interesting)

We tried the same recipe on arithmetic: is there an *add* arrow, a *multiply* arrow? **No.**
The "arrows" for math operations come out tangled with the numbers they apply to and cannot be
transplanted. That fits what other researchers have been finding: these models do not do math
with clean rules, but with a **bag of memorized tricks**. That the method can tell where the
clean structure exists and where it does not is exactly what makes it credible.

## Why does this matter?

These systems are used more and more, and almost nobody knows **how** they decide what to say.
This kind of work — it is called *interpretability* — is popping the hood: understanding the
internal parts is the first step toward auditing them, fixing them, and trusting (or
distrusting) them for reasons. Here we show that, at least for factual questions, there are
internal parts with clean structure: one piece for the thing, one piece for the question — and
they can be separated and transplanted.

## What this work does NOT claim

- It does not say the model "understands" or "thinks" like a person. We are talking about
  geometry: points and arrows in a space of numbers.
- We tested 5 question types, in English, over 12 countries, on 3 mid-sized models. The world
  is bigger than that, and the paper says so.
- Pushing the arrows has a cost: the operation changes the question precisely, but it also
  disturbs other things around it. We measured and reported that cost too.

---

**Want to see it with your own eyes?** The [interactive explorer](explorer.md) lets you move
the arrows yourself and watch the answer change. The full details, with every number and every
control, are in the [paper](assets/paper.pdf) and in [Evidence & controls](robustness.md).
