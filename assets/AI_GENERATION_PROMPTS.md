# AI Image Generation Prompts for Fugusashi Launch
**Model Recommendations**: Midjourney v6 / DALL-E 3 / Flux.1 (best for text/diagrams)
**Aspect Ratios**: Twitter 16:9 (1200×675), LinkedIn 1.91:1 (1200×628), Instagram/TikTok 9:16 (1080×1920), Square 1:1 (1080×1080)

---

## 1. TWITTER THREAD HERO — 1200×675 (16:9)
**File**: `assets/twitter_hero.png`
**Prompt**:
```
Professional tech announcement graphic for "Fugusashi v1.3.0" — open source LLM router.
Clean dark theme (background #0d1117), accent cyan (#00d4ff) and magenta (#ff006e).
Center: "Fugusashi" in modern geometric sans-serif (Inter/IBM Plex Sans), subtitle "Like Sakana Fugu. But Free."
Below: 3 key metrics in pill cards with icons: "80.0% Accuracy" (target icon), "22ms CPU" (cpu icon), "$0 Cost" (dollar-slash icon).
Bottom row: tech tags "ModernBERT • Federated • CMA-ES • MIT" in rounded rectangles.
Style: GitHub/technical blog aesthetic, high contrast, crisp, publication-ready. 1200x675.
```

**Midjourney**: `--ar 16:9 --style raw --stylize 250 --v 6.0`

---

## 2. BENCHMARK CHART STANDALONE — 1200×800 (3:2)
**File**: `assets/benchmark_chart.png`
**Prompt**:
```
Clean horizontal bar chart for technical blog post.
Title: "Routing Accuracy: Fugusashi vs Baselines"
Dark background (#0d1117), white gridlines, cyan/magenta accent bars.
Bars descending:
• Federated (3 clients, preliminary): 85% — BOLD CYAN with asterisk footnote "*20 hand-curated prompts"
• ModernBERT held-out (n=30): 80% — BOLD MAGENTA
• ModernBERT test (n=45): 80% — magenta
• Cost-only (cheapest): 36.7% — muted gray
• Random: 33.3% — muted gray
X-axis: 0-100%. Annotation arrow from 36.7% to 80.0%: "2.2× lift over cost-only" in orange.
Professional, publication-ready, IEEE style. 1200x800.
```

**Midjourney**: `--ar 3:2 --style raw --stylize 200 --v 6.0`

---

## 3. ARCHITECTURE DIAGRAM SIMPLIFIED — 1080×1080 (1:1)
**File**: `assets/arch_square.png`
**Prompt**:
```
Simplified architecture diagram for social media (Instagram/Twitter square).
Two tiers visually separated by dashed line.

TIER 1 (top, cyan theme): 4 strategy boxes horizontal — [Cost] [Similarity] [ModernBERT Learned] [CMA-ES]
→ merge arrow → "Confidence Gate" diamond → if <0.3 escalate down (orange arrow)

TIER 2 (bottom, magenta theme): Multi-agent flow
[Planner] → splits to 3 parallel: [Code Specialist] [Reasoning Specialist] [Creative Specialist]
→ converge at [Synthesizer] → "Final Answer"

Minimal text, icon-heavy (router, brain, gear, merge, split icons).
Dark bg (#0d1117), cyan tier 1, magenta tier 2, orange escalation.
Clean motion-graphics style. 1080x1080.
```

**Midjourney**: `--ar 1:1 --style raw --stylize 300 --v 6.0`

---

## 4. FEDERATED LOOP ANIMATION FRAMES — 4 frames × 1080×1920 (9:16)
**Files**: `assets/fed_frame_1.png` through `assets/fed_frame_4.png`
**Prompt** (generate 4 variations with same seed):

**Frame 1**:
```
Three laptop icons labeled "Client A", "Client B", "Client C", each with "Local CMA-ES" badge and lock icon (local data).
Dark background, cyan accent. Text: "Each client runs CMA-ES locally on their workload. Zero data leaves device."
Vertical 9:16 format for TikTok/Reels.
```

**Frame 2**:
```
Same three laptops, each emitting encrypted arrow (🔒 lock on arrow) toward central server icon "FedAvg + DP Noise (σ=0.1)".
Arrows labeled "DP-noised gradients". Server has shield icon.
Text: "Clients submit differentially-private gradients. Prompts never leave device."
```

**Frame 3**:
```
Server broadcasts "Global Router v2" back to all three clients via clean arrows.
Clients receive update, show "Router Updated ✓" with checkmark.
Text: "FedAvg aggregates → global model improves for everyone."
```

**Frame 4**:
```
All three clients show "Accuracy: 85%" with shield "Privacy Preserved" and small footnote "*preliminary: 20 hand-curated prompts".
Confetti/subtle celebration particles. Text: "85% routing accuracy (preliminary). 3 clients. Zero data shared."
Bottom: "Fugusashi — Federated Learning for LLM Routing"
```

**Midjourney**: `--ar 9:16 --style raw --stylize 250 --v 6.0 --seed 12345` (same seed for consistency)

---

## 5. EVOLUTION → DISTILLATION DIAGRAM — 1200×675 (16:9)
**File**: `assets/evolution_distillation.png`
**Prompt**:
```
Two-panel diagram showing "Evolution Teaches, Distillation Serves".

LEFT PANEL (Cyan theme): "Offline Evolution (CMA-ES)"
• Population of routing policies evolving over generations
• Fitness = benchmark accuracy
• Arrow: 100 generations → "Best Policy Weights"
• Label: "Runs nightly on your workload. Slow but thorough."

RIGHT PANEL (Magenta theme): "Online Inference (ModernBERT)"
• Single ModernBERT-base (149M) forward pass
• Input: prompt → Output: model class probabilities
• Latency: 22ms CPU
• Label: "Serves production traffic. Fast. Distilled from evolution."

CENTER BRIDGE: "Knowledge Distillation" — arrow from Left weights → Right model training.
Formula: L = CE(ModernBERT(prompt), CMA-ES_labels) + λ·KL(ModernBERT || CMA-ES_softmax)

Dark tech aesthetic. 1200x675.
```

**Midjourney**: `--ar 16:9 --style raw --stylize 250 --v 6.0`

---

## 6. TIKTOK/REELS THUMBNAIL — 1080×1920 (9:16)
**File**: `assets/tiktok_thumbnail.png`
**Prompt**:
```
Vertical thumbnail for TikTok/Reels/Shorts.
Split screen: LEFT (red tint) "Me paying $500/mo for GPT-4" — sad dev at laptop, money burning.
RIGHT (green/cyan tint) "Me using Fugusashi" — happy dev, terminal showing "Routed to gpt-oss-120b (87%)", confetti.
Center vertical text: "FREE FUGU 🍣🔪" in bold pufferfish emoji style.
Bottom: "80% accuracy • 22ms • $0" in pill badges.
High contrast, emotional, click-worthy. 1080x1920.
```

**Midjourney**: `--ar 9:16 --style raw --stylize 400 --v 6.0`

---

## 7. LINKEDIN HERO — 1200×628 (1.91:1)
**File**: `assets/linkedin_hero.png`
**Prompt**:
```
LinkedIn article hero image for "Why We Open-Sourced a $30M Idea".
Clean professional gradient: dark navy (#0a0e17) to slightly lighter (#111827).
Center: "Fugusashi" in elegant serif (Playfair Display) or clean sans (Inter Bold).
Subtitle: "The Open-Source LLM Router That Learns Your Workload"
Three metric cards horizontal:
[80.0% Accuracy] [22ms Latency] [MIT Licensed]
Bottom: "ModernBERT • Federated Learning • CMA-ES Evolution"
Subtle eulogik.com watermark bottom right.
Corporate-tech aesthetic, trustworthy, executive-friendly. 1200x628.
```

**Midjourney**: `--ar 1.91:1 --style raw --stylize 200 --v 6.0`

---

## 8. PRODUCT HUNT THUMBNAIL — 240×240 (1:1)
**File**: `assets/ph_thumbnail.png`
**Prompt**:
```
Tiny square thumbnail for Product Hunt (240x240).
Pufferfish (fugu) emoji style 🐡 but made of circuit traces / neural network nodes.
Cyan glow on dark background.
Text: "Fugusashi" tiny below.
Must be readable at 60x60. Icon-only preferred.
Style: App icon aesthetic, memorable, distinctive.
```

**Midjourney**: `--ar 1:1 --style raw --stylize 300 --v 6.0`

---

## 9. YOUTUBE THUMBNAIL — 1280×720 (16:9)
**File**: `assets/youtube_thumbnail.png`
**Prompt**:
```
YouTube thumbnail for "Build Your Own Fugu in 15 Minutes".
Split: LEFT "Sakana Fugu $30M" (red X, money burning), RIGHT "Fugusashi $0" (green check, terminal happy).
Center large text: "15 MIN BUILD" in bold yellow/black.
Bottom: "ModernBERT • 22ms • Free Models" in cyan pills.
Face: excited dev pointing at terminal (optional).
High CTR style: bright, contrast, readable at 10% size. 1280x720.
```

**Midjourney**: `--ar 16:9 --style raw --stylize 400 --v 6.0`

---

## 10. DEV.TO SERIES COVERS — 1000×420 (each)
**Files**: `assets/devto_part1.png`, `assets/devto_part2.png`, `assets/devto_part3.png`

**Part 1 Prompt**:
```
Cover for "From Prompt to Model: Inside a Learned LLM Router".
Dark bg, cyan accent. Center: ModernBERT architecture schematic (12 layers, attention heads).
Text: "Part 1: Fine-tuning ModernBERT for Routing" small.
Icon: transformer blocks → classification head → model class.
Technical, developer-focused. 1000x420.
```

**Part 2 Prompt**:
```
Cover for "Federated Learning for LLMs: Privacy-Preserving Router Evolution".
Dark bg, magenta accent. Center: 3 nodes → encrypted arrows → server → broadcast back.
Text: "Part 2: FedAvg + DP for Router Weights" small.
Shield/lock icons prominent. 1000x420.
```

**Part 3 Prompt**:
```
Cover for "CMA-ES + ModernBERT: When Evolution Meets Gradient Descent".
Dark bg, orange/yellow accent. Left: CMA-ES population evolving. Right: ModernBERT single pass.
Bridge: "Distillation" label with loss formula.
Text: "Part 3: Evolutionary Teacher, Gradient Student" small. 1000x420.
```

---

## BATCH GENERATION COMMANDS

### Midjourney (Discord) - Run sequentially:
```
/imagine [PROMPT_1] --ar 16:9 --style raw --stylize 250 --v 6.0
/imagine [PROMPT_2] --ar 3:2 --style raw --stylize 200 --v 6.0
/imagine [PROMPT_3] --ar 1:1 --style raw --stylize 300 --v 6.0
/imagine [PROMPT_4_FRAME_1] --ar 9:16 --style raw --stylize 250 --v 6.0 --seed 12345
/imagine [PROMPT_4_FRAME_2] --ar 9:16 --style raw --stylize 250 --v 6.0 --seed 12345
/imagine [PROMPT_4_FRAME_3] --ar 9:16 --style raw --stylize 250 --v 6.0 --seed 12345
/imagine [PROMPT_4_FRAME_4] --ar 9:16 --style raw --stylize 250 --v 6.0 --seed 12345
/imagine [PROMPT_5] --ar 16:9 --style raw --stylize 250 --v 6.0
/imagine [PROMPT_6] --ar 9:16 --style raw --stylize 400 --v 6.0
/imagine [PROMPT_7] --ar 1.91:1 --style raw --stylize 200 --v 6.0
/imagine [PROMPT_8] --ar 1:1 --style raw --stylize 300 --v 6.0
/imagine [PROMPT_9] --ar 16:9 --style raw --stylize 400 --v 6.0
/imagine [PROMPT_10_P1] --ar 1000:420 --style raw --stylize 250 --v 6.0
/imagine [PROMPT_10_P2] --ar 1000:420 --style raw --stylize 250 --v 6.0
/imagine [PROMPT_10_P3] --ar 1000:420 --style raw --stylize 250 --v 6.0
```

### DALL-E 3 (API) - Use exact prompts above with `size` parameter:
- 16:9 → `1792x1024` (closest)
- 3:2 → `1792x1024` crop
- 1:1 → `1024x1024`
- 9:16 → `1024x1792`
- 1.91:1 → `1792x1024` crop

### QUICK START - Generate All with Single Script

```bash
# Save prompts to files, then use DALL-E 3 batch or Midjourney
# For Midjourney: copy each prompt to Discord with params
# For DALL-E 3: use OpenAI API with prompts above

# Recommended order of priority:
# 1. twitter_hero.png (Day 0 thread)
# 2. benchmark_chart.png (Day 0 Reddit/thread)
# 3. arch_square.png (Day 1 LinkedIn/Reddit)
# 4. tiktok_thumbnail.png (Day 0 TikTok)
# 5. fed_frames_1-4.png (Day 2 TikTok)
# 6. youtube_thumbnail.png (Day 3 YouTube)
# 7. linkedin_hero.png (Day 1 LinkedIn)
# 8. ph_thumbnail.png (Day 7 Product Hunt)
# 9. evolution_distillation.png (Day 11 Dev.to Part 3)
# 10. devto covers (Days 4, 11, 18)
```

---

## BRAND ASSETS (Already in Repo)
| Asset | Location | Status |
|-------|----------|--------|
| Architecture Diagram (Fig 1) | `assets/fig1_architecture.png` | ✅ Ready |
| Benchmark Chart (Fig 2) | `assets/fig2_benchmark_results.png` | ✅ Ready |
| Paper PDF | `paper/main.pdf` | ✅ Ready |
| Model Card | `hf-models/fugusashi-v1.3/README.md` | ✅ Ready |
| Logo/Tagline | Create from prompts above | 🎨 Generate |

---

## COLOR PALETTE (Consistent Across All)
```css
--bg-dark: #0d1117;        /* GitHub dark */
--bg-card: #161b22;
--cyan: #00d4ff;           /* Primary: learned router, Tier 1 */
--magenta: #ff006e;        /* Secondary: orchestration, Tier 2 */
--orange: #ff6b00;         /* Alert: escalation, evolution */
--green: #00ff88;          /* Success: federated, accuracy */
--gray-muted: #8b949e;     /* Baselines, disabled */
--white: #ffffff;
--text-primary: #e6edf3;
--text-secondary: #8b949e;
```

## FONTS
- **Headers**: Inter Bold / IBM Plex Sans Bold / Space Grotesk
- **Body**: Inter Regular / IBM Plex Sans / JetBrains Mono (code)
- **Display**: Playfair Display (LinkedIn only) / Orbitron (tech headers)

## ICONS (Use Lucide/Phosphor/Tabler)
- Router: `route` / `git-branch`
- Brain/ML: `brain` / `cpu` / `network`
- Evolution: `git-merge` / `zap` / `trending-up`
- Federated: `shield` / `lock` / `users` / `share-2`
- Speed: `zap` / `gauge` / `timer`
- Cost: `dollar-sign` / `coin` / `tag`
- Accuracy: `target` / `check-circle` / `award`