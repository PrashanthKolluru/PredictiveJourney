# 🔗 Multi-Touch Marketing Attribution with Markov Chains 🚀

This project analyzes marketing channel performance using a Markov chain–based attribution model.

It takes in a single dataset (`Data.csv`) containing user paths through different marketing channels (like `eta`, `iota`, `alpha`, etc.), and produces:
- Incremental contribution of each channel to conversion
- How much conversion would drop if a channel didn’t exist
- A per-channel attribution share
- (Optionally) efficiency metrics if spend is provided

The goal is to answer:  
**"Which channels actually matter, and by how much?"**

---

## 📁 Dataset Description (`Data.csv`)

`Data.csv` is the only required input.

Each row represents one user journey.  
A journey is a sequence of touchpoints across channels, ending in either conversion or no conversion.

Typical columns in the dataset:

- `path`  
  The ordered sequence of channels the user touched, for example:  
  `eta > iota > alpha > eta`  
  This represents the marketing journey the user took.

- `converted`  
  A binary flag (0 or 1) telling us if this path ended in a successful conversion.
  
### Channel Names

In this dataset, the channels are abstracted as Greek-letter-style names.  
Some of the channels that appear in user paths include:

- `alpha`  
- `beta`  
- `delta`  
- `epsilon`  
- `eta`  
- `gamma`  
- `iota`  
- `kappa`  
- `lambda`  
- `mi`  
- `theta`  
- `zeta`

These act as marketing channels (e.g. email, paid social, search, display) but anonymized.  
For example, `eta` might represent "paid social", `iota` might represent "email", etc. The notebook treats them as channels without needing to know the real label.

---

## 🔄 Process Overview

Below is the full pipeline the notebook runs to turn `Data.csv` into channel attribution and efficiency metrics.

### 1️⃣ Parse Journeys
- Read each row of `Data.csv`.
- Split the `path` string (like `eta > iota > alpha > eta`) into an ordered list of touchpoints for that user.
- Clean up the sequence:
  - Remove immediate repeats so `eta > eta > iota` becomes `eta > iota`.  
    This prevents a channel from being over-credited just because it touched the user multiple times in a row.

This gives us a clean, minimal version of each marketing journey.

### 2️⃣ Add Start and End States
For each cleaned journey:
- Add a `"START"` state at the beginning.
- Add an ending state based on whether the user converted:
  - `"CONVERSION"` if `converted = 1`
  - `"NULL"` if `converted = 0` (meaning they dropped out without converting)

Example:
- Original path in the CSV:  
  `eta > iota > alpha > eta`
- If `converted = 1`, we transform it into:  
  `START → eta → iota → alpha → eta → CONVERSION`
- If `converted = 0`, we transform it into:  
  `START → eta → iota → alpha → eta → NULL`

Now we have full journeys that start the same way and always end in a terminal outcome.

This structure is important because it lets us analyze how users actually move across channels before buying (or not buying).

### 3️⃣ Build Channel Transition Behavior
From all journeys:
- Count how often users move from one state to the next.
  - e.g. `START → eta`
  - `eta → iota`
  - `iota → alpha`
  - `alpha → CONVERSION`
- Treat every unique channel (`alpha`, `eta`, `iota`, etc.) plus `START`, `CONVERSION`, and `NULL` as states in the system.

This gives us a behavioral model of how users tend to move through the funnel.

At this point we understand things like:
- Which channels tend to appear early (often right after `START`)  
- Which channels tend to show up right before `CONVERSION`  
- Which channels hand off to which other channels

This is already more informative than “last touch wins.”

### 4️⃣ Baseline Conversion Performance
Using all original journeys with all channels included, we measure the baseline performance of the funnel:
- How often do users eventually end in `CONVERSION` instead of `NULL`?
- Which sequences are most common on the way to `CONVERSION`?

This baseline represents:  
**"How the world looks with all channels (eta, iota, alpha, etc.) available."**

We use this baseline as a reference to judge the impact of each individual channel.

### 5️⃣ Channel Removal Simulation (Incremental Lift)
This is the core of the attribution logic.

For each channel in the dataset (for example, `eta`):
1. We simulate a “world without that channel.”
   - That means we rebuild every journey, but completely remove `eta` from the path.
   - Example:
     - Original converted path: `START → eta → iota → alpha → CONVERSION`
     - After removing `eta`: `START → iota → alpha → CONVERSION`
2. After removing that channel everywhere, we re-check how many journeys now end in `CONVERSION` vs `NULL`.

Then, we compare:
- Conversions in the original world (all channels present)
- Conversions in the “channel removed” world

The drop in conversions tells us how *critical* that channel was.

We do this for every channel:
- Remove `eta`, measure the damage
- Remove `iota`, measure the damage
- Remove `alpha`, measure the damage
- …and so on for all channels (`beta`, `gamma`, `theta`, etc.)

The result of that simulation is the **incremental lift** of each channel.

Plain English:
- If deleting `eta` causes a huge drop in conversion, `eta` is important.
- If deleting `iota` barely changes things, `iota` wasn’t really driving incremental value.

This is sometimes called the **removal effect** of a channel.

### 6️⃣ Attribution Share per Channel
Once we know how much each channel mattered (based on how much conversion suffers when we remove it), we turn that into a set of weights.

For each channel (like `eta`, `iota`, `alpha`, …):
- We calculate its contribution as a percentage of total incremental value across all channels.

Example of a possible output:
- `eta`   → 28% of incremental conversions  
- `iota`  → 22%  
- `alpha` → 18%  
- `gamma` → 12%  
- `theta` → 8%  
- `zeta`  → 5%  
- etc.

These percentages answer:
> “Out of all the marketing lift we’re getting, how much of it is thanks to this channel?”

This is our channel attribution.

### 7️⃣ Conversions Assigned Back to Each Channel
Now we take those attribution shares and apply them back to the actual number of conversions in the dataset.

Example:
- Suppose `Data.csv` represents 10,000 user journeys.
- Out of those, 1,200 journeys ended in `CONVERSION`.

If `eta` is responsible for 28% of incremental lift, we assign:
- `eta` → 0.28 × 1,200 ≈ 336 conversions

If `iota` is responsible for 22% of incremental lift, we assign:
- `iota` → 0.22 × 1,200 ≈ 264 conversions

This lets you report numbers your stakeholders understand:
- “Channel `eta` drove ~336 incremental conversions in this period.”
- “Channel `iota` drove ~264 incremental conversions.”
- “Channel `theta` barely moved conversions and is a budget cut candidate.”

This step converts theory into concrete headcounts.

### 8️⃣ (Optional) Efficiency / ROAS-like Metric
If you have spend by channel, you can extend the analysis into cost efficiency.

For each channel:
- Take its `attributed_conversions`.
- Divide by that channel’s spend in the period.
- Get a `conversions_per_dollar` score.

Example:
- If `eta` drove ~336 conversions and cost $4,000 in spend:
  - `eta` efficiency = 336 / 4000 = 0.084 conversions per $1
- If `iota` drove ~264 conversions and cost $1,500:
  - `iota` efficiency = 264 / 1500 = 0.176 conversions per $1

This makes it obvious which channels are expensive vs efficient, even if both look “important.”

---

## 📈 Final Output / What You Deliver

The notebook produces a channel-level summary that can be shown directly to Marketing, Growth, or Finance.  
For each channel (e.g. `eta`, `iota`, `alpha`, `beta`, …) you get:

- `channel`  
  The channel name from the dataset.

- `incremental_lift` / `removal_effect`  
  How much overall performance dropped when we removed that channel from all journeys.  
  Interprets as: “How crucial is this channel to driving conversions?”

- `attribution_share`  
  The percentage of incremental conversions that belong to this channel.

- `attributed_conversions`  
  How many conversions we credit to this channel in plain numbers, based on its share.

- (optional) `spend`  
  How much was spent on this channel in the time period.

- (optional) `conversions_per_dollar`  
  A ROAS-like efficiency metric. Higher = more efficient.

This table is what you put on slides when you say:
- “eta and iota are high-value channels.”
- “theta is expensive but not impactful.”
- “zeta assists but doesn’t close — keep it, but don’t overspend.”
- “We can safely trim channel X without breaking total conversion.”

---

## 🧠 Why This Approach Is Better Than Last Touch

- **Last-touch / first-touch lies.**  
  If `iota` always shows up at the end of the journey, last-touch will over-credit it.  
  If `eta` is always the first awareness touch, first-touch will over-credit it.  
  Neither actually tells you what happens if you *remove* that channel.

- **This method tests removal.**  
  It’s “What breaks if we delete you from the system?”  
  That’s much closer to incremental value.

- **It works with anonymous channels.**  
  Your dataset uses abstract channels like `eta`, `iota`, `alpha`, etc.  
  Even without knowing what `eta` “really” is in the marketing org, you can still quantify its importance and defend or cut its budget.

---

## 🔥 Conclusion

- We load `Data.csv`, which includes journeys through channels like `eta`, `iota`, `alpha`, `theta`, etc.
- We rebuild each journey from start to finish, including whether it converted.
- We simulate a world *without* each channel to measure how badly conversion suffers.
- That drop becomes the channel’s incremental contribution.
- We then assign conversion credit and (optionally) compute cost efficiency.
- `eta` and `iota` aren’t just steps in a path — we can now say how much they actually matter.
- You can use this to defend budget, cut waste, and explain channel value to non-technical stakeholders.
