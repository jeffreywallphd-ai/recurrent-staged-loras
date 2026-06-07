# Stage Split Validation

## Background

The project uses a three-stage decomposition for mathematical reasoning tasks:

* **Stage 1:** Problem
* **Stage 2:** Reasoning
* **Stage 3:** Final Answer

During inspection of the generated staged data, several examples appeared to have unintuitive boundaries between Stage 2 and Stage 3. The goal of this investigation was to identify the cause of these issues and improve the stage-splitting logic.

---

## Problem Identified

The original implementation extracted answers using heuristics such as:

* `####`
* `\boxed{...}`
* `"the answer is"`

For `\boxed{...}` answers, the splitter used the position of the `\boxed` token as the boundary between reasoning and answer.

Example:

### Original Response

```text
t = 3(28) + 4 = 84 + 4 = \boxed{88}
The answer is: 88
```

### Previous Stage Split

**Stage 2**

```text
t = 3(28) + 4 = 84 + 4 =
```

**Stage 3**

```text
88
```

This resulted in incomplete equations and reasoning traces.

---

## Additional Failure Mode

Nested boxed expressions were not handled correctly.

Example:

```text
\boxed{\sqrt{5}}
```

was extracted as:

```text
\sqrt{5
```

because the regular expression stopped at the first closing brace.

---

## Implemented Fixes

### 1. Preserve Complete Reasoning

Instead of cutting Stage 2 at the location of `\boxed{...}`, the splitter now:

1. Extracts the answer from the boxed expression.
2. Preserves the full reasoning text.
3. Replaces boxed expressions with their contents inside Stage 2.

Example:

### Updated Stage Split

**Stage 2**

```text
t = 3(28) + 4 = 84 + 4 = 88
```

**Stage 3**

```text
88
```

---

### 2. Support Nested Boxed Expressions

A brace-aware parser was implemented for extracting boxed values.

Example:

```text
\boxed{\sqrt{5}}
```

now correctly produces:

**Stage 2**

```text
Therefore, Gracie and Joe's points are \sqrt{5} units apart.
```

**Stage 3**

```text
\sqrt{5}
```

---

### 3. Remove Duplicate Final-Answer Markers

Additional cleanup was added to remove duplicated answer markers such as:

```text
The answer is:
Answer:
Final Answer:
```

from reasoning text when appropriate.

---

## Validation

Validation was performed using an inspection script over MetaMathQA examples.

Observed improvements:

* Complete equations are preserved in Stage 2.
* Reasoning sentences are no longer truncated at boxed expressions.
* Nested boxed expressions are extracted correctly.
* Stage 3 retains clean answer targets.
* Dataset generation and training pipelines continue to execute successfully after the changes.

A debug training run completed successfully after the modifications, indicating compatibility with the existing training pipeline.

---

## Conclusion

The updated splitter produces more semantically coherent Stage 2 reasoning traces while preserving clean Stage 3 answer targets. The modifications improve supervision quality for staged reasoning experiments and address several common failure modes observed in the original implementation.
