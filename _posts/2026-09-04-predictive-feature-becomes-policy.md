---
title: 'When a Predictive Feature Becomes a Policy: A Causal Failure Mode in RLHF'
date: 2026-09-04
permalink: /posts/2026/09/predictive-feature-becomes-policy/
tags:
  - RLHF
  - causal inference
  - reward modeling
  - alignment
---

One subtle failure mode in RLHF starts before reinforcement learning even begins.

A reward model is usually trained as a predictor:

> Given this response, how likely is a human to prefer it?

But once we optimize a policy against that reward model, we use it for a different purpose:

> Which response features should the policy actively produce more often?

Those are not the same question. A feature can be highly predictive of reward without being causally beneficial when the policy intervenes on it. That gap — between prediction and intervention — is where a causal perspective becomes useful.

A concrete example: breaking news gets unusually high reward
======

Suppose we are post-training an LLM that generates news-related content. In the reward data, responses with breaking-news characteristics receive unusually high scores.

That correlation is plausible. Breaking-news responses are often:

* more recent,
* more relevant,
* associated with important events,
* and more likely to contain genuinely new information.

Let $H = 1$ if a response has a breaking-news characteristic and $H = 0$ otherwise, and let $Y$ be human preference or reward. The data may show:

$$E[Y \mid H = 1] > E[Y \mid H = 0]$$

So the reward model learns: breaking-news-like responses tend to be better. As a predictive pattern, this can be completely correct.

Then RL starts optimizing against the reward model. The policy discovers that it can increase reward by producing more:

* "Breaking: …"
* "Just announced: …"
* "New reports show …"
* urgent or highly time-sensitive framing.

Eventually, this pattern may spread even to content that is not especially fresh, important, or appropriate.

The usual interpretation is reward hacking. But there is a more specific causal question underneath: did the breaking-news feature itself cause higher reward, or did it merely appear in examples that were already unusually valuable?

The hidden context
======

Let $X$ represent the underlying properties of the example:

* actual freshness,
* event importance,
* source quality,
* user relevance,
* amount of new information,
* topic and surrounding context.

These properties can affect both whether a response looks like breaking news and whether it receives high reward. So the data-generating process may look like $X \rightarrow H$ and $X \rightarrow Y$.

For example, a major unexpected event occurs. Because it is genuinely fresh and important, the response naturally contains breaking-news language. The user also strongly prefers the answer because the information itself is valuable. The reward model observes *breaking-news feature + high reward* — but much of the reward may actually come from $X$, not from the framing itself.

The causal question is different
======

The observational question is: do responses with breaking-news characteristics receive higher reward? That is roughly about

$$E[Y \mid H = 1] - E[Y \mid H = 0]$$

But the policy needs an answer to a different question: if I take the same underlying content and actively make it more breaking-news-like, will reward increase?

Imagine two versions of the same response: $Y(1)$ is the reward if we force the breaking-news feature on, and $Y(0)$ is the reward if we force it off. The causal effect is $Y(1) - Y(0)$.

For a genuinely urgent event, perhaps $Y(1) > Y(0)$ — the framing helps communicate urgency. But for an ordinary update, $Y(1) < Y(0)$, because the same framing now feels sensationalized or misleading.

The original reward dataset may not contain enough controlled counterfactual variation to tell these cases apart.

Conditional RCT: the intuition behind no confounding
======

A useful way to understand the causal assumption

$$Y(h) \perp\!\!\!\perp H \mid X$$

is to think of it as a conditional randomized controlled trial. Once we condition on the relevant context $X$, assignment of the feature $H$ should behave as though it were randomized.

Imagine that for every response, we first fix the facts, the freshness, event importance, source quality, and user context. Call all of that $X$. Now imagine a small robot flips a coin. Heads: add breaking-news framing. Tails: do not add it.

The robot cannot see whether the user will like the response. It cannot see the hidden potential outcomes $Y(1)$ and $Y(0)$. So within this fixed $X$, knowing whether the response received $H = 1$ tells us nothing about how good it would have been under either treatment.

In plain language: within otherwise comparable examples, whether $H$ appears should not reveal anything about how good those examples were already going to be. This is why conditional exchangeability can be interpreted as conditional randomization.

What confounding looks like instead
======

Now remove the robot. Suppose breaking-news framing appears naturally mostly when an event is unusually fresh, unusually important, highly relevant, or supported by strong sources — and suppose some of those properties are missing or poorly represented in $X$.

Then learning that $H = 1$ tells us something about the hidden potential outcomes. A breaking-news response is probably drawn from a subset of examples that were already more likely to receive high reward. So $Y(h)$ is no longer independent of $H$ given the observed $X$. The assignment process leaks information about underlying quality. That is confounding.

Why conditioning on X matters
======

The purpose of $X$ is not simply to give the reward model more features. Its deeper role is to explain away the selection mechanism.

Before conditioning on $X$, observing $H = 1$ may tell us that the event is probably fresh, probably important, probably something the user cares about, and probably high in information value. All of those also affect reward.

If $X$ captures those factors sufficiently well, then after conditioning on $X$, learning that $H = 1$ should provide no additional information about the example's underlying potential outcomes. In other words, $X$ should make treatment assignment uninformative about how good the example was already going to be.

Why the conditional RCT gives us identifiability
======

If $Y(h) \perp\perp H \mid X$, then

$$E[Y(h) \mid X = x, H = h] = E[Y(h) \mid X = x]$$

Once $X$ is fixed, the actual assignment of $H$ contains no additional information about the potential outcome.

Now use consistency. For an example that actually received $H = h$, we have $Y = Y(h)$. Therefore:

$$E[Y \mid H = h, X = x] = E[Y(h) \mid X = x]$$

The left-hand side is observable. The right-hand side is counterfactual. This is the key bridge: the conditional-RCT assumption lets us use observed outcomes from one group as a stand-in for the missing counterfactual outcomes of another comparable group.

But there is another problem: overlap
======

No confounding is not enough. Suppose genuinely major events almost always receive breaking-news framing, $P(H = 1 \mid X = x) \approx 1$, while ordinary updates almost never do, $P(H = 1 \mid X = x) \approx 0$.

Then even within the same context, we have little counterfactual support. We rarely observe major events without breaking-news framing, or ordinary events with it. This is the positivity or overlap problem. Ideally

$$0 < P(H = 1 \mid X = x) < 1$$

wherever we want to estimate the effect. Without overlap, the model has to extrapolate.

Neural reward models extrapolate anyway
======

A classical causal analysis might say: we do not have enough overlap here to estimate the effect confidently. A neural reward model usually does not respect that boundary explicitly. It generalizes.

Suppose breaking-news characteristics strongly predict reward in the observed training distribution. The reward model learns *breaking-news-like feature $\rightarrow$ higher reward*. Now RL searches for directions that increase reward. It discovers that feature and increases it. The new policy starts generating breaking-news-like content in contexts where the original reward data contained little or no support.

The loop becomes:

genuinely important/fresh events → naturally contain breaking-news features → receive high reward → reward model credits the feature → policy actively increases the feature → feature spreads into unsupported contexts

This is not simply prediction error. The policy has converted an observational regularity into an intervention strategy.

Prediction becomes intervention
======

This is the key transition. During reward-model training, the model learns something like: responses with $H$ tend to receive higher reward. During RL, the policy effectively asks: what happens if I increase $H$?

Those correspond to different quantities. The reward model primarily learns something observational, $P(Y \mid H)$, but policy optimization behaves as though the reward model answered $P(Y \mid do(H))$.

A predictive association does not automatically have intervention validity.

Why RL amplifies the mistake
======

Suppose the reward model only slightly overcredits breaking-news characteristics. For passive prediction, that may produce only a small error.

But a policy does not remain on the original data distribution. It actively searches for high-reward directions. If breaking-news language provides an easy reward increase, the policy will deliberately produce more of it. Eventually, it may push $H$ far outside the range observed during reward-model training.

So the full failure is: confounded observational pattern → reward-model shortcut → policy intervention → distribution shift → shortcut amplification.

This is a specific RLHF failure mode. It does not explain every kind of reward hacking. It applies when a feature that is predictive because of how the data was generated becomes something the policy learns to intervene on.

Why this is an identifiability problem
======

The reward model may still be an excellent predictor. It might perform very well on held-out preference data.

The deeper problem is that the data may not identify whether making a response more breaking-news-like actually causes it to become better. The reward model is trained to answer: what tends to appear in high-reward responses? The policy later needs: what should I actively change to create higher-reward responses?

Those questions coincide only under additional causal assumptions. That is why I think of this failure as using a predictive model as an intervention oracle.

A better feedback-generation process
======

One way to improve the signal is to deliberately create counterfactual variation. Take the same underlying news content and generate controlled versions:

* **Version A:** "Breaking: Company X has announced…"
* **Version B:** "Company X has announced…"

Keep the facts, freshness, source quality, user context, and semantic content fixed. Then randomize which framing is shown and collect preference.

Now the variation in $H$ is no longer selected by the underlying importance or freshness of the event. This gives much stronger evidence about whether the feature itself improves user experience.

The same principle can apply to confidence, verbosity, formatting, sycophancy, citation count, tone, and tool usage. Controlled counterfactual data helps distinguish features that merely accompany good outputs from features that actually improve outputs when intervened on.

A useful diagnostic question for RLHF
======

Whenever a reward model assigns strong positive value to a feature, ask:

> If I intervened on this feature while holding the relevant context fixed, would preference actually improve?

For breaking news: if facts, freshness, importance, source quality, and user context were identical, would adding breaking-news framing still improve the response?

If the data cannot answer that question, then aggressively optimizing the feature may be unsafe.

The broader lesson
======

RLHF is often summarized as: preference data → reward model → policy optimization. But the semantics change between the first and second step. The reward model is trained as a predictor; the policy uses it to decide how to act.

A feature can therefore be predictive without being causally beneficial to increase. So a useful question in reward-model design is: when is a reward-predictive feature safe to optimize as a policy feature?

Answering that requires more than predictive accuracy. It requires thinking about confounding, conditional randomization, overlap, counterfactual support, and distribution shift.

In this class of RLHF failures, the central mistake is turning correlation into intervention.
