---
title: 'When a Predictive Feature Becomes a Policy: A Causal Failure Mode in RLHF'
date: 2026-09-04
permalink: /posts/2026/09/predictive-feature-becomes-policy/
tags:
  - RLHF
  - causal inference
  - reward modeling
  - alignment
excerpt: "One subtle failure mode in RLHF starts before reinforcement learning even begins."
---

<div class="notice" markdown="1">
#### TL;DR

- A reward model is trained to answer *what tends to appear in high-reward responses?* The policy then uses it to answer *what should I produce more of?* Those are different questions.
- A feature can be strongly predictive purely through confounding: hidden context — freshness, importance, source quality — drives both the feature and the reward.
- Conditioning on that context only helps if it actually explains the selection, and only where treated and untreated examples overlap. Reward models extrapolate past that boundary anyway, and RL actively pushes them there.
- The fix is controlled counterfactual variation: hold the content fixed, vary the feature, and measure whether preference actually improves.
</div>

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

<figure style="display:block">
<div style="overflow-x:auto">
<svg viewBox="0 0 720 340" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Two causal diagrams. In the observational panel, hidden context X points to both the breaking-news feature H and the reward Y, and H points to Y. The X-to-H and X-to-Y arrows form a backdoor path that makes H and Y correlated without H causing Y. In the interventional panel, the arrow from X to H is cut, because the policy sets H itself; only the causal H-to-Y edge and the X-to-Y edge remain." style="width:100%;min-width:560px;height:auto;font-family:system-ui,-apple-system,'Segoe UI',sans-serif">
<defs>
<marker id="f1-ab" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#2a78d6"/></marker>
<marker id="f1-ao" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#eb6834"/></marker>
<marker id="f1-ag" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#c3c2b7"/></marker>
<marker id="f1-ad" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#52514e"/></marker>
</defs>
<line x1="360" y1="20" x2="360" y2="292" stroke="#e1e0d9" stroke-width="1"/>
<text x="175" y="36" text-anchor="middle" font-size="11.5" fill="#898781">hidden context</text>
<line x1="162" y1="95" x2="116" y2="177" stroke="#eb6834" stroke-width="2.2" marker-end="url(#f1-ao)"/>
<line x1="188" y1="95" x2="234" y2="177" stroke="#eb6834" stroke-width="2.2" marker-end="url(#f1-ao)"/>
<line x1="126" y1="205" x2="218" y2="205" stroke="#2a78d6" stroke-width="2.6" marker-end="url(#f1-ab)"/>
<circle cx="175" cy="72" r="26" fill="#f0efec" stroke="#52514e" stroke-width="2"/>
<circle cx="100" cy="205" r="26" fill="#ffffff" stroke="#52514e" stroke-width="2"/>
<circle cx="250" cy="205" r="26" fill="#ffffff" stroke="#52514e" stroke-width="2"/>
<text x="175" y="79" text-anchor="middle" font-size="17" font-weight="600" fill="#0b0b0b">X</text>
<text x="100" y="212" text-anchor="middle" font-size="17" font-weight="600" fill="#0b0b0b">H</text>
<text x="250" y="212" text-anchor="middle" font-size="17" font-weight="600" fill="#0b0b0b">Y</text>
<text x="100" y="250" text-anchor="middle" font-size="11.5" fill="#898781">feature</text>
<text x="250" y="250" text-anchor="middle" font-size="11.5" fill="#898781">reward</text>
<text x="175" y="282" text-anchor="middle" font-size="12.5" fill="#52514e">(a) What the reward model learns</text>
<text x="545" y="36" text-anchor="middle" font-size="11.5" fill="#898781">hidden context</text>
<line x1="532" y1="95" x2="486" y2="177" stroke="#c3c2b7" stroke-width="2.2" stroke-dasharray="5 4" marker-end="url(#f1-ag)"/>
<line x1="501" y1="128" x2="517" y2="144" stroke="#d03b3b" stroke-width="2.6" stroke-linecap="round"/>
<line x1="517" y1="128" x2="501" y2="144" stroke="#d03b3b" stroke-width="2.6" stroke-linecap="round"/>
<line x1="558" y1="95" x2="604" y2="177" stroke="#52514e" stroke-width="2.2" marker-end="url(#f1-ad)"/>
<line x1="496" y1="205" x2="588" y2="205" stroke="#2a78d6" stroke-width="2.6" marker-end="url(#f1-ab)"/>
<circle cx="545" cy="72" r="26" fill="#f0efec" stroke="#52514e" stroke-width="2"/>
<circle cx="470" cy="205" r="26" fill="#ffffff" stroke="#52514e" stroke-width="2"/>
<circle cx="620" cy="205" r="26" fill="#ffffff" stroke="#52514e" stroke-width="2"/>
<text x="545" y="79" text-anchor="middle" font-size="17" font-weight="600" fill="#0b0b0b">X</text>
<text x="470" y="212" text-anchor="middle" font-size="17" font-weight="600" fill="#0b0b0b">H</text>
<text x="620" y="212" text-anchor="middle" font-size="17" font-weight="600" fill="#0b0b0b">Y</text>
<text x="470" y="250" text-anchor="middle" font-size="11.5" fill="#898781">set by policy</text>
<text x="620" y="250" text-anchor="middle" font-size="11.5" fill="#898781">reward</text>
<text x="545" y="282" text-anchor="middle" font-size="12.5" fill="#52514e">(b) What the policy does: do(H)</text>
<line x1="118" y1="314" x2="146" y2="314" stroke="#2a78d6" stroke-width="2.6"/>
<text x="154" y="318" font-size="11.5" fill="#52514e">causal effect</text>
<line x1="248" y1="314" x2="276" y2="314" stroke="#eb6834" stroke-width="2.2"/>
<text x="284" y="318" font-size="11.5" fill="#52514e">backdoor path (confounding)</text>
<line x1="470" y1="314" x2="498" y2="314" stroke="#c3c2b7" stroke-width="2.2" stroke-dasharray="5 4"/>
<line x1="480" y1="309" x2="488" y2="319" stroke="#d03b3b" stroke-width="2.2" stroke-linecap="round"/>
<line x1="488" y1="309" x2="480" y2="319" stroke="#d03b3b" stroke-width="2.2" stroke-linecap="round"/>
<text x="506" y="318" font-size="11.5" fill="#52514e">edge removed by do(H)</text>
</svg>
</div>
<figcaption><strong>Figure 1.</strong> The same three variables, two different questions. In (a), the reward model sees <em>H</em> and <em>Y</em> move together — but part of that association travels the orange backdoor path <em>H</em> ← <em>X</em> → <em>Y</em>, which carries no causal effect at all. In (b), the policy <em>sets</em> <em>H</em> rather than observing it, which severs <em>X</em> → <em>H</em>. Note that no orange remains: <em>X</em> still causes <em>Y</em>, but with the arrow into <em>H</em> gone there is no longer a backdoor path for it to travel. Only the blue edge survives, so only the blue edge is what optimizing <em>H</em> actually buys. The gap between the panels is the gap between <em>P</em>(<em>Y</em> | <em>H</em>) and <em>P</em>(<em>Y</em> | do(<em>H</em>)).</figcaption>
</figure>

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

Balance is the same concern stated quantitatively. Overlap is a yes-or-no identification condition — does every context admit both treatments at all? Balance asks how *close* the treated and control distributions actually are, $p(x \mid H = 1)$ against $p(x \mid H = 0)$, which is what governs whether the effect can be estimated well from a finite sample. Positivity can hold in the strict sense while $e(x)$ sits at 0.01 in one region and 0.99 in another; the estimand exists, and we still have almost nothing to estimate it with.

Stating it that way exposes a lever. It is tempting to treat overlap as a fixed fact about the dataset — either the groups look alike or they do not — but the comparison is always made *in some representation*, and the representation is ours to choose.

This is the idea behind balanced representation learning. Instead of conditioning on a hand-specified feature vector $X$ and hoping it captures the selection mechanism, learn a representation $\Phi(X)$ under two objectives simultaneously: predict the outcome well from $\Phi$ and the treatment indicator, and keep the treated and control distributions in $\Phi$-space close together. The second objective is what recovers counterfactual support the raw features never had.

<figure style="display:block">
<div style="overflow-x:auto">
<svg viewBox="0 0 720 300" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Two scatter panels of treated and control points. In the left panel, raw context space, control points occupy the left region and treated points the right, with only a narrow band in the middle where both appear; outside that band each region contains only one group, so no counterfactual comparison is available. In the right panel, the learned representation, the two groups are mixed throughout, so every region contains both treated and control points." style="width:100%;min-width:560px;height:auto;font-family:system-ui,-apple-system,'Segoe UI',sans-serif">
<rect x="190" y="40" width="60" height="150" fill="#1baf7a" fill-opacity="0.13"/>
<rect x="400" y="40" width="250" height="150" fill="#1baf7a" fill-opacity="0.13"/>
<rect x="70" y="40" width="250" height="150" rx="5" fill="none" stroke="#e1e0d9" stroke-width="1.5"/>
<rect x="400" y="40" width="250" height="150" rx="5" fill="none" stroke="#e1e0d9" stroke-width="1.5"/>
<text x="135" y="32" text-anchor="middle" font-size="10.5" fill="#898781">control only</text>
<text x="220" y="32" text-anchor="middle" font-size="10.5" fill="#52514e">both</text>
<text x="282" y="32" text-anchor="middle" font-size="10.5" fill="#898781">treated only</text>
<text x="525" y="32" text-anchor="middle" font-size="10.5" fill="#52514e">both, everywhere</text>
<g fill="#eb6834">
<circle cx="90" cy="168" r="4.2"/><circle cx="145" cy="111" r="4.2"/><circle cx="87" cy="75" r="4.2"/><circle cx="130" cy="117" r="4.2"/><circle cx="91" cy="64" r="4.2"/><circle cx="88" cy="90" r="4.2"/><circle cx="184" cy="139" r="4.2"/><circle cx="165" cy="57" r="4.2"/><circle cx="219" cy="134" r="4.2"/><circle cx="122" cy="130" r="4.2"/><circle cx="135" cy="119" r="4.2"/><circle cx="137" cy="60" r="4.2"/><circle cx="141" cy="56" r="4.2"/><circle cx="156" cy="110" r="4.2"/><circle cx="82" cy="117" r="4.2"/><circle cx="156" cy="133" r="4.2"/><circle cx="82" cy="110" r="4.2"/><circle cx="132" cy="87" r="4.2"/><circle cx="244" cy="158" r="4.2"/><circle cx="130" cy="141" r="4.2"/><circle cx="122" cy="88" r="4.2"/><circle cx="155" cy="61" r="4.2"/><circle cx="136" cy="159" r="4.2"/><circle cx="98" cy="101" r="4.2"/><circle cx="196" cy="52" r="4.2"/><circle cx="115" cy="78" r="4.2"/>
</g>
<g fill="#2a78d6">
<circle cx="284" cy="176" r="4.2"/><circle cx="232" cy="102" r="4.2"/><circle cx="295" cy="150" r="4.2"/><circle cx="273" cy="86" r="4.2"/><circle cx="278" cy="173" r="4.2"/><circle cx="268" cy="148" r="4.2"/><circle cx="271" cy="65" r="4.2"/><circle cx="269" cy="60" r="4.2"/><circle cx="258" cy="122" r="4.2"/><circle cx="232" cy="108" r="4.2"/><circle cx="272" cy="69" r="4.2"/><circle cx="303" cy="133" r="4.2"/><circle cx="278" cy="79" r="4.2"/><circle cx="276" cy="86" r="4.2"/><circle cx="308" cy="90" r="4.2"/><circle cx="241" cy="163" r="4.2"/><circle cx="260" cy="160" r="4.2"/><circle cx="285" cy="133" r="4.2"/><circle cx="308" cy="79" r="4.2"/><circle cx="308" cy="85" r="4.2"/><circle cx="256" cy="89" r="4.2"/><circle cx="222" cy="61" r="4.2"/><circle cx="290" cy="83" r="4.2"/><circle cx="276" cy="128" r="4.2"/><circle cx="226" cy="173" r="4.2"/><circle cx="279" cy="113" r="4.2"/>
</g>
<g fill="#eb6834">
<circle cx="542" cy="161" r="4.2"/><circle cx="453" cy="71" r="4.2"/><circle cx="617" cy="155" r="4.2"/><circle cx="468" cy="76" r="4.2"/><circle cx="579" cy="170" r="4.2"/><circle cx="456" cy="172" r="4.2"/><circle cx="611" cy="128" r="4.2"/><circle cx="507" cy="65" r="4.2"/><circle cx="421" cy="173" r="4.2"/><circle cx="466" cy="141" r="4.2"/><circle cx="470" cy="156" r="4.2"/><circle cx="547" cy="89" r="4.2"/><circle cx="452" cy="143" r="4.2"/><circle cx="428" cy="81" r="4.2"/><circle cx="538" cy="159" r="4.2"/><circle cx="551" cy="87" r="4.2"/><circle cx="619" cy="78" r="4.2"/><circle cx="416" cy="86" r="4.2"/><circle cx="513" cy="60" r="4.2"/><circle cx="452" cy="98" r="4.2"/><circle cx="541" cy="69" r="4.2"/><circle cx="494" cy="164" r="4.2"/><circle cx="634" cy="135" r="4.2"/><circle cx="568" cy="126" r="4.2"/><circle cx="444" cy="56" r="4.2"/><circle cx="416" cy="167" r="4.2"/>
</g>
<g fill="#2a78d6">
<circle cx="570" cy="173" r="4.2"/><circle cx="417" cy="132" r="4.2"/><circle cx="521" cy="144" r="4.2"/><circle cx="484" cy="178" r="4.2"/><circle cx="429" cy="121" r="4.2"/><circle cx="579" cy="165" r="4.2"/><circle cx="579" cy="141" r="4.2"/><circle cx="591" cy="167" r="4.2"/><circle cx="492" cy="138" r="4.2"/><circle cx="616" cy="162" r="4.2"/><circle cx="506" cy="152" r="4.2"/><circle cx="607" cy="124" r="4.2"/><circle cx="553" cy="100" r="4.2"/><circle cx="544" cy="129" r="4.2"/><circle cx="430" cy="133" r="4.2"/><circle cx="636" cy="163" r="4.2"/><circle cx="577" cy="101" r="4.2"/><circle cx="578" cy="125" r="4.2"/><circle cx="512" cy="158" r="4.2"/><circle cx="431" cy="147" r="4.2"/><circle cx="419" cy="128" r="4.2"/><circle cx="521" cy="81" r="4.2"/><circle cx="570" cy="115" r="4.2"/><circle cx="551" cy="168" r="4.2"/><circle cx="470" cy="53" r="4.2"/><circle cx="480" cy="137" r="4.2"/>
</g>
<defs>
<marker id="f2-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#52514e"/></marker>
</defs>
<line x1="332" y1="115" x2="386" y2="115" stroke="#52514e" stroke-width="2" marker-end="url(#f2-a)"/>
<text x="358" y="105" text-anchor="middle" font-size="11.5" fill="#898781">learn Φ</text>
<text x="195" y="218" text-anchor="middle" font-size="15" fill="#0b0b0b">Context <tspan font-style="italic">x</tspan></text>
<text x="195" y="239" text-anchor="middle" font-size="11.5" fill="#898781">support only in a narrow band</text>
<text x="525" y="218" text-anchor="middle" font-size="15" fill="#0b0b0b">Representation <tspan font-style="italic">Φ(x)</tspan></text>
<text x="525" y="239" text-anchor="middle" font-size="11.5" fill="#898781">support throughout</text>
<circle cx="256" cy="274" r="4.5" fill="#2a78d6"/>
<text x="268" y="278" font-size="11.5" fill="#52514e">treated (H = 1)</text>
<circle cx="390" cy="274" r="4.5" fill="#eb6834"/>
<text x="402" y="278" font-size="11.5" fill="#52514e">control (H = 0)</text>
</svg>
</div>
<figcaption><strong>Figure 2.</strong> Overlap is not only a property of the data — it is partly a property of the space you measure it in. In raw context space the two groups occupy largely separate regions, and only the shaded band contains both; everywhere else a response has no comparable counterpart under the opposite treatment, so one potential outcome is simply never observed. A representation learned under a balance penalty mixes the groups, restoring the comparisons the raw features never supported. The dot-cloud convention follows Johansson, Shalit and Sontag, <a href="https://arxiv.org/abs/1605.03661">Learning Representations for Counterfactual Inference</a> (ICML 2016).</figcaption>
</figure>

Why a learned representation beats a feature list
======

This matters more than it might appear, because it changes what we are betting on.

Writing down $X$ by hand — freshness, importance, source quality, user relevance — is a bet that we enumerated the confounders correctly. Every property that drives both the framing and the reward, and that we failed to think of, stays in the error term and keeps leaking selection information into $H$. The failure is silent: the model fits well, and the treatment-effect estimate is quietly wrong.

A learned $\Phi(X)$ relaxes that bet in two ways. It can extract confounding structure from raw inputs — the full response, the retrieved documents, the conversation state — that no hand-written feature list would have named. And because it is high-dimensional and continuous rather than a short list of tabular attributes, it has room to represent the many weak, interacting factors that actually determine whether a response looks like breaking news. Conditioning on a dozen engineered features is a coarse approximation of that structure; conditioning on a learned representation is a much finer one.

The catch is that this only works if the representation is regularized toward balance. A representation trained purely for outcome accuracy has no reason to make the groups comparable — in fact it may do the opposite, since separating treated from control examples can be an easy way to lower prediction loss. The imbalance term is what keeps the representation honest, and it is exactly the term a standard reward model does not have.

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
