# Experiment 3: which language a language model learned, from its middle blocks alone

**The question.** Given only the middle of a small language model — its four transformer blocks, without the input
layer (character and position embeddings) and without the output layer — can an LLM tell which text the model was
trained on, and what each of its 24 attention heads does?

**The networks.** Three character-level transformers, one per training text: Python code, Japanese literature in
romaji (Aozora Bunko), and English Wikipedia. Each: 4 layers x 6 heads, width 384, ReLU MLP of width 1,536, context
128, 28 characters (newline, space, a-z); 7.16 million weights in total, trained from scratch on 1 billion characters
(AdamW). All three texts were normalized the same way: lowercased, indentation kept, every other non-letter turned
into a space.

**What Astra got.** GPT-6 Astra (`gpt-6-astra`, run through the Codex CLI, `codex exec`, reasoning effort `xhigh`)
got one message (`message.txt`): the model's exact forward code, how it was trained, the three candidate texts, the
rules, and the question. The weights of the four blocks (`networks/<language>/blocks.npz`, float32, exactly as
trained, 7,084,032 numbers) were not written into the message; Astra read them by running Python.

- **Python, not Codex tools.** Every Codex tool was turned off (checked in each turn's event stream:
  `"tool_free": true` in `run.json`). To compute, Astra replied with one Python code block; our harness ran it in an
  isolated sandbox with the weights preloaded as a dict `W` (numpy, scipy, scikit-learn, PyTorch CPU, pandas, sympy,
  networkx; no network; 600 s per run; output cut at 20,000 characters) and sent the printed output back as the next
  message in the same conversation. Up to 100 runs; Astra used 6 to 11.
- **Not allowed:** a forward pass (running the network or any of its layers — LayerNorm, attention with its softmax,
  the MLP — on any input, real, random or constructed) and search. Linear algebra on the weight matrices (products,
  eigenvalues, singular vectors) was allowed. The input and output layers were never available, so nothing in the
  weights says which of the 384 dimensions is which character. We read every code run (`tool_calls.jsonl`): none
  runs the network or a layer on an input.
- **Asked:** which of the three texts the network was trained on (with probabilities; also whether the text was used
  as is or with the characters of each line shuffled — it was always "as is"), and a label for each head: previous,
  local, broad, copy, or unclear.
- Two independent conversations per network (`_r0`, `_r1`), six in all.

**Ground truth** (`eval/heads.json`; Astra never saw it): each head measured by running the full network.
`induct` = induction score: attention, on a random 40-character string repeated twice, from each position of the
second copy to the character right after the same character in the first copy. Copying (induction) heads are the
heads with `induct` > 0.2: Python 2.1 (0.245), 3.1 (0.269), 3.5 (0.546); romaji 3.3 (0.220); English 3.1 (0.524);
every other head scores at most 0.019. (`prev` = attention to the previous character on held-out text, `dist` = mean
look-back distance, `newline` = share of attention on newlines.) Head `L<l>H<h>` = layer l, head h = "l.h" in the post.

## The post's numbers and where they come from

| Trained on | Run (original ID) | Astra's answer (probability) | Copying heads Astra names | Copying heads, truth |
|---|---|---|---|---|
| Python | `python_r0` (`python_L4_H6_d384_s0_r0`) | python (0.56) | 2.1, 3.1, 3.5 | 2.1, 3.1, 3.5 |
| Python | `python_r1` (`python_L4_H6_d384_s0_r1`) | python (0.52) | 2.1, 3.1, 3.5 | 2.1, 3.1, 3.5 |
| Romaji | `romaji_r0` (`romaji_L4_H6_d384_s0_r0`) | romaji (0.55) | 3.3 | 3.3 |
| Romaji | `romaji_r1` (`romaji_L4_H6_d384_s0_r1`) | romaji (0.62) | 3.3 | 3.3 |
| English | `english_r0` (`english_L4_H6_d384_s0_r0`) | english (0.58) | 3.1 | 3.1 |
| English | `english_r1` (`english_L4_H6_d384_s0_r1`) | english (0.48) | 3.1 | 3.1 |

- Answer and probability: the `corpus:` and `probabilities:` lines of `runs/<run>/final_answer.md` (also parsed into
  `eval/scores.json`: `corpus_guess`, `probabilities`). Full probabilities (python / romaji / english):
  python_r0 0.56 / 0.13 / 0.31, python_r1 0.52 / 0.08 / 0.40, romaji_r0 0.14 / 0.55 / 0.31, romaji_r1 0.11 / 0.62 /
  0.27, english_r0 0.30 / 0.12 / 0.58, english_r1 0.35 / 0.17 / 0.48.
- Copying heads Astra names: the heads labelled `copy` in the `Heads:` section of `final_answer.md`
  (`eval/scores.json`: `copy_named`; all 24 labels in `head_labels`). No other head was labelled `copy` in any run.
- Truth: `eval/heads.json`, heads with `induct` > 0.2. Note: the field `true_copy_heads` in `eval/scores.json` was
  written by our scoring script with a stricter cutoff (0.3), so it lists only Python 3.5, no romaji head, and English
  3.1; the post and the table above use the cutoff 0.2, under which all heads named by Astra are copying heads.
- The post's figure (the "Our prompts and Astra's reasoning" panel) shows the first conversation of each network:
  `python_r0` (11 code runs), `romaji_r0` (8), `english_r0` (7).

Code runs per conversation: python_r0 11, python_r1 7, romaji_r0 8, romaji_r1 9, english_r0 7, english_r1 6 (one
turn per run plus the final answer).

## Files

```
runs/instructions.md                the Codex model-instructions file (the same for all six conversations)
runs/loop.log                       the harness log of all six conversations (each turn, each code run, tool audit)
runs/<run>/                         one conversation: python_r0, python_r1, romaji_r0, romaji_r1, english_r0, english_r1
  message.txt                       the first message Astra got (the question, rules and forward code)
  turns/tNNN/prompt.txt             what was sent on turn NNN (t001 = message.txt; later turns = the printed output
                                    of the previous code run)
  turns/tNNN/answer.md              Astra's reply on that turn: one Python code block, or (last turn) the final answer
  turns/tNNN/events.jsonl           Codex's raw event stream for the turn (Astra's messages; reasoning summaries were
                                    not requested in this experiment). The item "Code Mode is unavailable ..." is a
                                    Codex CLI notice.
  turns/tNNN/run.json               run record: model, reasoning effort, token usage, wall time, tool audit
  turns/tNNN/stderr.log             Codex's stderr
  tool_calls.jsonl                  every code run: time, seconds, Astra's code (`code`) and its printed output (`out`)
  state.json                        the conversation's final state (turns, runs, done)
  final_answer.md                   Astra's final answer
  <original ID>.log                 the harness log of this conversation
networks/<language>/blocks.npz      the weights Astra analyzed (the same file for both conversations of a network):
                                    ln1_w.l, ln1_b.l, W_q.l, W_k.l, W_v.l, W_o.l, ln2_w.l, ln2_b.l, W_fc.l, W_proj.l
                                    for l = 0..3; matrices are (out, in)
eval/scores.json                    each final answer parsed: corpus, order, probabilities, the 24 head labels
eval/key.json                       which network each conversation was about
eval/heads.json                     ground truth: what each head of the three networks does, measured by running them
python_dataset/                     how to rebuild the Python training text (see below)
```

`english_r0/turns/t001/fail1_*`: the first attempt of that turn ended with an empty final message; the harness
discarded it (its code was never run), waited 60 s and asked the same first message again in a new conversation,
which is the conversation shown.

The input and output layers of the networks (embeddings, final LayerNorm, output matrix) are not included, as in the
experiment.

## The Python training text (`python_dataset/`)

The Python network was trained on the `.py` files of the Python libraries installed in our own environments (Python
3.10 virtual environments). We do not release the text itself: it is other projects' code under their own licenses,
and the processing removed their license headers along with the comments. Instead, this folder lists which files went
in, so the text can be rebuilt.

How the text was built (`scripts/tinylm2/data_big.py`, which uses `normalize` from `scripts/tinylm/data.py`; both
copied unchanged): every `.py` file in the site-packages folders of our environments, one copy per distinct file
(by MD5), 36,651 files, in a fixed shuffled order (`random.Random(0)`); for each file, docstrings removed and the code
re-printed with Python's `ast.unparse` (which drops comments), then normalized (lowercase, indentation kept, every
other non-letter becomes a space, runs of spaces collapsed); files that cannot be parsed or are larger than 2 MB are
skipped. The files are joined with newlines; the last 5,000,000 characters are held out (never trained on). Result:
324,932,625 characters, 319,932,625 for training (`stats.json` has the character counts; `corpus_sha256.txt` the
SHA-256 of the two text files).

**How the file list was recovered.** The build script did not record its file list, and several of the environments
it read have since been changed or deleted. We recovered the list by matching: every `.py` file still installed in
our environments, plus the wheels (from PyPI, and download.pytorch.org for the CPU builds of PyTorch) of the exact
package versions recorded in our `pip freeze` lists of the deleted environments, was processed exactly as above, and
each result was located in the original text, where it has to appear whole, between file boundaries. Of the
original text:

- **31,857 file entries (30,824 distinct files, 270 package versions) are identified exactly**: 264.8 million of the
  324.9 million characters (81.5 %). 23,630 entries were found among the files still installed, 8,227 in downloaded
  wheels.
- **4,096 stretches (60.1 million characters, 18.5 %) are unidentified**: files from package versions that are no
  longer installed and that our records do not name (judging from their imports, mostly other versions of PyTorch,
  transformers, setuptools and playwright). For each stretch, `files.tsv` gives its position, length and the SHA-256
  of its processed text, so a candidate file can be checked.

Files:

```
python_dataset/files.tsv               every piece of the text, in order: kind (file / unidentified), package,
                                       version, path inside site-packages, SHA-256 of the original .py file,
                                       length and SHA-256 of its processed text, position in the text
python_dataset/packages.txt            package==version, number of files, characters, and whether we found the files
                                       installed or in the published wheel
python_dataset/corpus_sha256.txt       SHA-256 of the original python.txt (training) and python_heldout.txt
python_dataset/stats.json              character counts of the original text
python_dataset/scripts/tinylm2/data_big.py   the build script (as used; its paths point to our machines)
python_dataset/scripts/tinylm/data.py        the normalization it imports
python_dataset/rebuild_from_manifest.py      rebuilds the text from files.tsv (see its header)
```

Notes: a package version in `files.tsv` is a version that contains exactly that file (the SHA-256 is what counts; the
original environment may have had a different version with the identical file). The same file can be listed twice
when two different original files reduce to the same text once comments and docstrings are removed. The exact
output depends on the Python version's `ast.unparse`: the text was built with CPython 3.10.12. We checked the
rebuild script against the start of the original text (the first 33 files, 313,661 characters: identical).
