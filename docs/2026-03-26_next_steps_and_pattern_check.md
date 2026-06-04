# 2026-03-26 Next Steps and Pattern-vs-Planning Check

## Why This Document Exists

The current `state-only 1k / 8k` experiment improved holdout performance substantially:

- reached-goal accuracy increased to `0.47`
- answer completeness is now stable
- `<answer>` and `<grid_token>` both appear reliably

However, this still does not answer the more important question:

- did the model actually learn planning
- or did it mostly learn a stronger output pattern

This document records the recommended next steps and the diagnostic plan.

## 1. What the Current Result Suggests

The latest result supports two conclusions at the same time:

1. Data scale mattered.

- moving from 500 to 1000 training samples helped a lot
- increasing training steps also helped

2. Data scale is not the whole story.

- the model no longer mainly fails on formatting
- the main remaining failures are:
  - falling into holes
  - going out of bounds

So the bottleneck has shifted:

- before: unstable output structure
- now: unstable planning safety

## 2. Recommended Next-Step Directions

### A. Continue Scaling Data

The simplest next move is:

- expand from `1k` to `2k`
- then potentially to `5k`

But this should not be only a same-distribution scale-up.

Prefer increasing diversity along:

- hole density
- path length
- reachable vs unreachable ratio
- map topology variety
- later, map size

Reason:

- if the new data only repeats the same surface patterns, the model may just memorize better

### B. Strengthen Safety-Oriented Supervision

Current main errors are not:

- missing `<answer>`
- malformed structure

Current main errors are:

- unsafe actions
- hole entry
- out-of-bounds moves

So the next supervision improvements should emphasize:

- path executability
- local action safety
- consistency between intermediate reasoning and final action sequence

Practical options:

- increase the weight on final answer path tokens
- give extra penalty to trajectories that hit holes
- give extra penalty to trajectories that go out of bounds
- add step-level analysis to identify where the first unsafe action happens

### C. Test Generalization More Aggressively

A stronger model should survive distribution shifts.

Recommended OOD tests:

- train on `4x5`, test on `5x5` or `6x6`
- train on one hole-density regime, test on another
- train on shorter paths, test on longer paths
- test on different prompt wording for the same environment

If performance collapses under these shifts, the model is likely still relying heavily on memorized patterns.

### D. Reduce Template Overfitting in the CoT

Right now the generated CoT is strong but quite templated.

That is not automatically bad, but it can hide shallow learning.

Possible mitigation:

- vary the narration style in the training data
- diversify local phrasing for state checks
- allow multiple valid verbalizations for the same search process
- avoid a single dominant wording pattern for candidate comparison

Goal:

- make the model rely more on state and less on fixed narration templates

## 3. How To Tell Whether the Model Learned Planning or Just Patterns

This is the core diagnostic question.

Below are the most useful checks.

### A. OOD Generalization Check

This is the strongest test.

Method:

- train on one distribution
- evaluate on a shifted distribution

Examples:

- larger maps
- different hole density
- different path-length range
- different unreachable/reachable balance

Interpretation:

- if performance stays reasonably stable, that is evidence of genuine planning
- if performance collapses quickly, that is evidence of pattern dependence

### B. Same Map, Different Prompt Check

Method:

- keep the map identical
- rewrite the user instruction or system prompt slightly

Examples:

- different wording
- different sentence order
- different map-size phrasing

Interpretation:

- a true planner should keep producing a valid path
- a pattern-memorizing model may become unstable under superficial wording changes

### C. Counterfactual Map Edit Check

Method:

- keep almost everything the same
- change only one environmental fact

Examples:

- move one hole
- move the goal
- block one previously safe cell

Interpretation:

- if the output path adapts correctly, that suggests state-sensitive reasoning
- if the output stays close to the previous high-frequency template, that suggests memorization

### D. Intermediate-Reasoning / Final-Answer Consistency Check

This project is well positioned for this check because the model emits:

- `<grid_token>`
- intermediate state narration
- final `<answer>`

Method:

- inspect whether intermediate reasoning claims match the final action sequence

Examples of inconsistency:

- the reasoning says a direction contains a hole
- but the final answer later walks into that direction

Interpretation:

- if such contradictions are common, the model is likely generating plausible reasoning text without truly using it to plan

### E. Path Template Reuse Analysis

Method:

- collect all predicted answers on holdout
- count repeated path strings
- measure how many different maps share the same predicted path template

Interpretation:

- if a few path strings dominate many distinct maps, that is a sign of template bias
- if predictions vary in a state-sensitive way, that is a sign of real environmental conditioning

### F. Oracle Shortest-Path Comparison

Method:

- compute the true shortest safe path programmatically
- compare it with the model output

Metrics:

- success or failure
- path length gap
- first error step
- whether the chosen path is safe but suboptimal, or outright invalid

Interpretation:

- a memorized-template model often outputs plausible but non-adaptive paths
- a planning-capable model should stay closer to valid shortest paths, especially on easy maps

## 4. Most Useful Immediate Diagnostics

If only a few next analyses can be done, these should be first:

1. Holdout path-template repetition analysis

- detect whether many different samples collapse to a few common answer templates

2. Intermediate-reasoning vs final-answer consistency analysis

- detect whether the model's narrated state checks actually align with its chosen actions

3. OOD evaluation

- especially map-size shift or hole-density shift

These three together would give a much better answer to:

- did the model really learn planning
- or did it mainly learn a better-looking planning script

## 5. Current Working Hypothesis

The current evidence suggests a mixed picture:

- the model has definitely learned more than formatting
- it conditions on the map enough to improve substantially
- but it may still rely heavily on a small set of recurrent search-and-path templates

So the most likely state right now is:

- partial planning ability
- plus significant template memorization

The next experiments should be designed specifically to separate these two effects.

## 6. Bootstrap + DPO as a Possible Next Phase

Another realistic next step is:

- bootstrap new candidate solutions from the current model
- then train with DPO on automatically constructed preference pairs

This is feasible for this project because the environment is executable.

For each predicted answer, the pipeline can automatically check:

- whether it reaches the goal
- whether it falls into a hole
- whether it goes out of bounds
- whether the answer format is complete
- whether the path is shorter or longer

So the project is in a good position to build preference data from environment feedback instead of relying only on human preference annotation.

### Why It Is Reasonable Here

The current failure mode is no longer mainly:

- missing `<answer>`
- malformed structure

The current failure mode is mainly:

- unsafe paths
- hole entry
- out-of-bounds moves

DPO is attractive here because it can push the model toward:

- safer trajectories
- more executable answers
- stronger preference for successful plans over failed plans

This is more direct than continuing to rely only on token-level imitation.

### Recommended Bootstrap-to-DPO Pipeline

Step 1:

- start from the current SFT model
- sample multiple candidate answers for each training problem

Step 2:

- execute each predicted answer in the environment
- score each candidate using programmatic feedback

Step 3:

- build preference pairs:
  - chosen = better answer
  - rejected = worse answer

Step 4:

- run DPO using these automatically built preference pairs

### Suggested Ranking Criteria for Preference Construction

A good ranking should not use only one signal.

Recommended priority:

1. whether the path reaches the goal
2. whether the path is illegal:
   - hole
   - out of bounds
3. path length
4. answer completeness
5. consistency between reasoning and final answer

This is preferable to using only:

- reached goal / did not reach goal

because single-signal ranking can encourage shallow shortcuts.

### Main Risks

Bootstrap + DPO is not automatically beneficial.

The main risks are:

1. low-quality bootstrap candidates

- if most generated candidates are bad, the preference gap may be weak
- DPO then learns from poor supervision

2. reinforcing shallow templates

- if the chosen answers are selected only because they look cleaner
- the model may become more templated without becoming more intelligent

3. strengthening fake reasoning

- if full `<think>` traces are used without checking consistency
- DPO may reward text that sounds like reasoning but does not drive action quality

### Recommended Guardrails

To reduce these risks:

- always use environment execution as the primary ranking signal
- keep structure completeness as a secondary signal, not the main one
- add explicit reasoning-answer consistency checks where possible
- inspect a subset of preference pairs manually before full DPO training

### Current Recommendation

Bootstrap + DPO is worth trying in this project.

But the safer order is:

1. build the automatic evaluator
2. generate and inspect preference pairs
3. verify that chosen vs rejected pairs are truly high-quality
4. then run DPO

So the answer is:

- yes, this is a viable next stage
- and it is better suited here than generic preference tuning
- but only if preference construction is grounded in executable environment feedback
- but it may still rely heavily on a small set of recurrent search-and-path templates

So the most likely state right now is:

- partial planning ability
- plus significant template memorization

The next experiments should be designed specifically to separate these two effects.
