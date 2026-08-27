# Hugging Face collaborators — who to add, and who we could not find

Hugging Face has **no API for adding collaborators to a repo**. It is a web-UI action,
so this file is the click-list. Re-derive it any time with
`python3 tools/hf_confirm_users.py`.

## How these were matched — and why that matters

Each row was resolved by querying the public quicksearch endpoint
(`/api/quicksearch?q=<term>&type=user`) and keeping hits whose **fullname** matched the
colleague's name. **None of the twelve GitHub handles exists as a Hugging Face username** —
`tngchien`, `spsagar13`, `odincodeshen`, `dmalone-arm` all return nothing. The mapping is
by display name only.

> **A matching display name is not proof of identity.** Two people can share a name, and
> these repos are **private**. Adding the wrong `Na Li` hands a stranger the dataset.
> Confirm the username with each person over Arm email or Teams before you add them.
> The eight below are the ones worth confirming, not the ones already confirmed.

## Add these eight

| # | Colleague | Hugging Face username | Account fullname | Confidence |
|---|---|---|---|---|
| 1 | Alexander Tsyplikhin | `atsyplikhin` | Alexander Tsyplikhin | exact, sole hit |
| 2 | Darrell Malone | `darrellkmalone` | Darrell Malone | exact, sole hit |
| 3 | Odin Shen | `odinshen` | Odin Shen | exact, sole hit |
| 4 | Sagar Surendran | `sagarsurendran` | Sagar Surendran | exact, sole hit |
| 5 | Masoud Koleini | `koleinimasou` | masoud koleini | case-insensitive match |
| 6 | Jackie Lee | `JKLEE1015` | Yiwei Lee | GitHub handle → account whose fullname is his Chinese name |
| 7 | Dominica Amanfo | `DominicaAmanfo` | Dominica Abena Oforiwaa Amanfo | **two candidates** — see below |
| 8 | Timo Tang | `tngx` | timo tang | case-insensitive match |

### Row 7 is ambiguous
Two accounts could be her: `DominicaAmanfo` (fullname *Dominica Abena Oforiwaa Amanfo*)
and `Dominica26` (fullname *Dominica Amanfo*). `DominicaAmanfo` is used here because it was
named explicitly, but it is a coin-flip on the evidence alone. Ask her which is hers.

## Four we could not find

| Colleague | What the search returned |
|---|---|
| Belinda Wang | Nothing, under any spelling tried |
| Shaneil Parsad | Only `Shaneil Ming`, `shaneil chandran`, `Shaneillahi` — different people |
| Fei Xiang | `feixiang`, `feixiang-01`, `feixiang007` — fullnames *deniao*, *fx*, *wen*; none matches |
| Na Li | `nali`, `nali-88`, `nali000` — fullnames *nali Badran*, *nabeel ali*, *Numair Ali*; all different people |

These four need to send you their Hugging Face username directly, or create an account.
Do not guess from the near-misses above.

## Where to click

Three private repos hold this project's data. Each needs the collaborators added
separately — Hugging Face has no organisation-style inheritance between them here:

| repo | settings page |
|---|---|
| `armwaheed/go2-peer-detection` (dataset) | `https://huggingface.co/datasets/armwaheed/go2-peer-detection/settings` |
| `armwaheed/go2-peer-detector` (model) | `https://huggingface.co/armwaheed/go2-peer-detector/settings` |
| `armwaheed/mappo-quadruped-nav` (model) | `https://huggingface.co/armwaheed/mappo-quadruped-nav/settings` |

All three are **private** and must stay private: the corpus is Arm office footage
containing an identifiable person. The account also owns two public
`stable-diffusion-3.5-medium-onnx` models, which are unrelated to this project and are not
where any of this data goes.

Re-derive the list any time from a logged-in shell:

```bash
python3 -c "from huggingface_hub import HfApi; a=HfApi(); n=a.whoami()['name']; \
  [print(d.id) for d in a.list_datasets(author=n)]; [print(m.id) for m in a.list_models(author=n)]"
```
