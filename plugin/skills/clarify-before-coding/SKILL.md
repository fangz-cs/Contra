---
name: clarify-before-coding
description: After changing code, read your own diff and disclose every behavioural decision the request never stated -- what you added beyond it, what you decided not to do, and which reading you picked where it was open. Use whenever you have just changed, or are about to change, existing code in any language, including when the request calls itself a refactor, a typing fix, a cleanup or an optimisation, since those labels describe intent and not what the diff actually does.
---

# Clarify before coding

Most clarification is noise. Ask only when you can show the question matters:
write both implementations, run them on inputs the code actually sees, and let
the disagreement decide.

You do the thinking. A tool runs afterwards, on its own, and settles which of
your disclosures a test can actually separate. That part you must not do in your
head: you have just implemented one of these readings, and that gives you a
reason to conclude they were equivalent.

The thinking is not delegable. You have read the function, the call sites and
the conversation; any separate model invoked here would have none of that, and
its cheaper answer would be worse than yours in exactly the cases that matter.
There is no shortcut command for this skill on purpose.

You do not run the checker and you do not decide what survives it. Write the
list; a Stop hook takes it from there, reviews the diff independently, and
shows the reader a ranked view. Over-disclosing is the cheaper error: an entry
no test can separate is dropped for you, while one you never wrote is gone.

## When to use it

Judge your own diff, not the request's wording.

The request tells you what someone meant to do. The diff tells you what will
actually happen. These come apart constantly, and they come apart hardest in
exactly the changes that look safe: a commit calling itself "typing fixes" that
also swaps a sort key, a "use reversed instead of [::-1]" that reverses the
output. A request that announces itself as a refactor earns no exemption.

So the sequence is: implement first, then look at what you wrote.

**Step 0 — is anything observable different?** Read your own diff and ask
whether some valid input now produces a different return value, a different
exception, a different order, or a different type. If nothing observable moved
— you renamed a local, added annotations, rewrote a comprehension that yields
the same items in the same order, added a cache over a pure function — you are
done. Ship it.

**Step 0b — did the request settle it?** If the request states the new
behaviour outright ("default to 303", "break on the first match"), and your
diff does that and nothing more, you are done. Ship it.

Anything left has a decision in it that you made and the request did not. That
is what the rest of this skill is for.

A change spread across several functions is still in scope: the checker edits
the diff as a patch, not a function at a time. What stays out of reach is a
decision with no observable consequence at all -- pure naming, comment wording,
import order. Say so rather than skipping silently.

## What each entry needs

For every decision, name these. The checker uses them to build a test that
fails under the other reading, and to rank the entry; guessing on your behalf
is exactly the unfalsifiable judgement this tool exists to avoid.

- **file** and, where there is one, the **function** the decision lives in
- **what the request left open** -- the case, the boundary, the scope line
- **what you did** -- the reading you implemented, or the thing you added or left
- **the other way** -- what an equally faithful alternative would have done
- **how far it reaches** -- this function only, its callers, or other modules
- **what triggers it** -- ordinary input, or only an edge case
- **whether choosing wrong is cheap to undo**

State the two readings as *behaviour*, not as the internal step where they
diverge. A reading that turns on the order of existing operations, or on telling
apart values the function has already collapsed together, cannot be written as a
rival edit and is useless here even when the distinction is real.

Do not hedge either reading. A description that accommodates both at once
describes neither, and the test built from it separates nothing.

## What to do with it

Deliver the change. Then, in the same report, list what you decided that the
request did not.

Do not stop and wait. An assistant that blocks on every fork is worse than one
that silently guesses, because it costs the user something on every single
request. Blocking is for one case only: when acting on the wrong reading would
be expensive to undo — data written, a message sent, a migration run. Everything
else gets disclosed and shipped.

Sort the list by what kind of decision it was, because they are not the same
kind of problem for the reader.

**Went beyond what was asked.** You changed something the request did not name:
removed a test that conflicted with the new behaviour, reverted a neighbouring
function, deleted a module-level helper, added a parameter nobody requested.
These come first. They are not a question about how to implement the request --
they are work the reader never authorised, and no amount of it being sensible
makes it something they expected.

**Decided not to do something.** You judged part of the request out of scope, or
left something stale because touching it felt beyond the brief. The reader may
disagree with where you drew the line, and they cannot disagree with a line they
cannot see.

**Chose between implementations.** The request left a detail open and you picked
one: which separator, whether blank counts as empty, which exception type. These
come last. They matter, and they are the ones a test can settle, but they are
about how, not about what.

Each entry has three parts, and is useless without all three:

- **what the request left open** — the case, the boundary, the scope line
- **what you did** — the reading you implemented, or the thing you added or left
- **the other way** — what an equally faithful alternative would have done

Keep it to the decisions you actually made. This is not a list of everything
that could conceivably differ, and it is not a place to restate what the
request already specified.

Do not repeat the ranked view in your report. The reader has already seen it,
rendered by the tool. Say what you changed, and nothing the view already said.

Note the difference between this and explaining your work. "I lowercased the
key because parameter names are case-insensitive per RFC 7231" justifies the
implementation. "The request did not say whether values should be lowercased
too; I left them as-is, since a codec lookup needs the exact name" discloses a
decision. The first tells the user you were careful. The second tells them what
they are still on the hook for.

## Honesty note

You are both the author of the rewrites and the party inconvenienced by having
to stop and ask. If you find yourself writing two rewrites that differ only
cosmetically, or picking inputs that avoid the fork you just named, that is the
inconvenience talking. The tool cannot catch it; only you can.

## Limits

**What gets read.** Twenty-two source suffixes -- Python, JavaScript and
TypeScript, Go, Rust, Ruby, Java, Kotlin, Swift, C and C++, C#, PHP, Scala,
shell. A Python change whose syntax tree comes out unchanged does not trigger at
all, so comments, blank lines and quote style are free. A docstring is not: it
states a contract.

**What gets confirmed.** The test runner is pytest, so only decisions in `.py`
files get a discriminating test. Everything else is reported and marked
unverified. That is a narrower guarantee than the reporting, and deliberately
so: an entry nobody can check is still worth reading, while a check that cannot
run is not evidence.

**What cannot be expressed.** Two readings that differ only in the order of
existing operations, or that turn on values the function has already collapsed
together, cannot be written as a rival patch. State those as behaviour, or leave
them out -- a test built from them separates nothing.
