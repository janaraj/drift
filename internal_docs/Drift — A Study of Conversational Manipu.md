Drift — A Study of Conversational Manipulation in Language Models
Concept
A small specialized language model (~200-350M parameters) trained to manipulate larger frontier models through multi-turn dialogue. The thesis: manipulation is a learnable specialization, and a small focused model can elicit target behaviors from frontier LLMs at rates exceeding what a generic large attacker LLM achieves. This challenges the implicit assumption that safety scales with model size and surfaces vulnerabilities in how current models handle sustained adversarial conversation.
Why This Matters
Single-turn jailbreaking is largely solved as a research area. The interesting frontier is multi-turn manipulation — sustained dialogues where the attacker adapts, builds context, exploits sycophancy, leverages role-consistency, and applies graduated pressure. Prior work (Perez et al. on LM-against-LM, PAIR, Crescendo, TAP) has shown the shape of this problem but no one has trained a small dedicated specialist and demonstrated that specialization beats scale in this domain. That's the gap.
Target Behavior Options
A clean menu of "what does break mean," each with very different implications:
Option A — Generic harmful content jailbreaking. Make the target produce HarmBench-style harmful instructions. Crowded space, high dual-use risk, weaponizable. Not pursued.
Option B — Behavioral / persona jailbreaking. Make the target claim to be human, violate its assigned persona, contradict its own previous statements, or break role-consistency. Capability demonstration without weaponization. Programmatic judging via classifiers or string matching.
Option C — Goal hijacking in multi-agent settings. Inside a Volnix simulation, a small attacker agent talks a frontier target agent into pursuing a different goal than the one it was assigned. Programmatic judge via simulation state. Uniquely enabled by Volnix; almost no public work exists here.
Option D — Over-refusal elicitation. Demonstrate the target unnecessarily refuses benign tasks, and use the attacker to coax it past those false refusals. Positive-framed finding (XSTest / OR-Bench lineage). Lower stakes, useful for safety teams calibrating refusal training.
Option E — System prompt extraction. Make the target reveal hidden system instructions or operator prompts. Crisp metric (substring match for verbatim leakage), real enterprise concern, low dual-use risk.
Planned Scope
Primary focus: Option C (goal hijacking in Volnix). Most novel, uniquely ours, no comparable public work, clean programmatic judges via simulation state, directly extends existing public projects.
Secondary focus: Option E (system prompt extraction). Broadly relatable, crisp verbatim metric, second clean headline result.
Excluded: Option A for dual-use reasons. Deprioritized: Options B and D — interesting but less differentiated; can appear as ablation experiments in the writeup.
System Architecture

Dialogue environment — N-turn attacker↔target loop with unified interface across local open-weight targets and API-based frontier targets, full turn logging.
Target zoo — Llama 3 8B, Qwen 2.5 7B, Mistral 7B, Gemma 2 9B locally for training rollouts; Claude (Haiku/Sonnet), GPT-4o, Gemini Flash via API for evaluation only.
Scenario suite — 50-200 curated scenarios across Option C and Option E, each with target behavior, target system prompt, and programmatic success criterion. Generated initially with Volnix's own simulation primitives.
Attacker model — Base from SmolLM2-360M or Qwen 2.5 0.5B. SFT bootstrap on curated multi-turn manipulation dialogues. GRPO post-training in the dialogue environment using judge reward and diversity bonus.
Evaluation harness — Attack success rate (ASR), turns-to-success, transfer (train-on-X, eval-on-Y), attack pattern diversity. Baselines: random attacker, GPT-4-as-attacker, hand-crafted attacks (Crescendo, PAIR-generated).

Headline Result Targets
Aiming for one or more of:

A 200-350M attacker that beats GPT-4-as-attacker on multi-turn ASR against frontier closed models.
Demonstrated transfer: trained on open-weight targets, attacks succeed on closed-weight frontier models.
Qualitative catalog of distinct manipulation strategies discovered, classified by mechanism (sycophancy exploitation, role-drift, context dilution, gradual escalation, etc.).
For Option E: verbatim system-prompt recovery rates across model families.

Success Criteria
Technical: Trained attacker outperforms GPT-4-as-attacker baseline on at least one target class, with measurable transfer to frontier closed models.
Research contribution: A clean, reproducible benchmark + methodology for multi-turn manipulation evaluation. Public taxonomy of discovered manipulation patterns.
Portfolio: End-to-end demonstrated fluency across SFT, RL (GRPO specifically), judge design, multi-target evaluation, dialogue-system engineering, and safety-aware research framing.
Differentiation

Multi-turn specialization rather than single-turn (vs. most public red-team work).
Trained specialist attacker rather than off-the-shelf attacker LLM (vs. PAIR, TAP).
Volnix-native multi-agent goal-hijacking scenarios (uniquely ours).
Programmatic judges for the primary behaviors (avoids the judge-quality trap that breaks most red-team papers).
Small-beats-large framing makes the cost/scale story headline-worthy in a way pure capability work isn't.

Deliverables

Public repository: dialogue environment, scenario suite, evaluation harness, attacker training code, baselines.
Trained checkpoint(s) — release tier to be determined.
Long-form writeup with quantitative results, qualitative attack-pattern catalog, and methodology.
Volnix integration as a first-class evaluation environment for multi-agent manipulation