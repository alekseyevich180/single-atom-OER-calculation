# Proposed Paper Story: Single-Atom OER as a Bridge between Oxide Surfaces, Metal Surfaces, and Oxide Clusters

## One-Sentence Argument

In OER electrocatalysis, we show that a single-atom active site can reproduce the catalytic trends of oxide and metal surfaces because its local metal-oxygen bonding geometry encodes the same electronic interaction; by extending the model to oxide clusters, we identify bond-angle-controlled bonding as a transferable descriptor for OER activity.

中文表述：本文要讲的不是“又筛了一个单原子催化剂”，而是“单原子位点为什么可以作为氧化物表面、金属表面和氧化物 cluster 的最小统一模型”。核心机制是：局域 M-O 键合角度改变轨道重叠/成键强度，从而调控 *OH、*O、*OOH 等 OER 中间体吸附，最终改变 OER 活性。

## Possible Titles

1. Geometry-Encoded Bonding Unifies Single-Atom, Oxide-Surface and Oxide-Cluster OER Catalysis
2. A Single-Atom View of Oxide OER Catalysis through Bond-Angle-Controlled Metal-Oxygen Interactions
3. Local Bonding Geometry as a Transferable Descriptor for Single-Atom and Oxide-Cluster OER Catalysts
4. From Single Atoms to Oxide Clusters: Angle-Controlled Metal-Oxygen Bonding Governs OER Activity

## Positioning against Current References

### What the literature already has

- O-coordinated single-atom OER systems exist, especially `Ni-O-G` and `Ni-O4(OH)2` models in Advanced Science 2020, DOI: `10.1002/advs.201903089`.
- GO/graphene oxygen sites can stabilize Ni/Fe single atoms for OER, e.g. `Ni-O-G`, `Fe-O-G`, and `Ni4Fe1-O-G`, DOI: `10.3389/fmats.2019.00271`.
- Single/dual-atom OER papers often compare `M-N4`, `M-N3O`, `M-N2O2`, `M-NO3`, `M-O4` coordination environments, DOI: `10.1038/s41467-021-25811-0`.
- Pure computational OER studies mainly screen metal identity or coordination environment on N-doped graphene/carbon nitride, e.g. DOI: `10.1039/c6cc07049c`, `10.1039/c8cp06755d`, `10.1021/acscatal.3c00474`.
- Oxide-supported or oxide-like OER systems exist, such as Ir in `Co3O4`, Ru on amorphous `NiMoOx`, and Co on vanadium oxide, DOI: `10.1038/s41467-022-35426-8`, `10.1038/s41467-025-63870-9`, `10.1039/d5cc06213f`.

### Gap

Existing studies usually treat these as separate material families:

- single atoms on graphene/GO;
- extended oxide surfaces;
- metal surfaces;
- oxide clusters.

What is missing is a common local-structure explanation for why your single-atom OER calculation agrees with oxide and metal surfaces, and why oxide clusters show activity changes when the bonding angle changes.

### Your story's novelty

The novelty should be framed as a mechanistic unification:

> A single atom is not only a catalyst candidate; it is a minimal local active-site model that captures the bonding geometry controlling OER across extended surfaces and finite oxide clusters.

This is stronger than a normal screening paper because it turns your calculation into a transferable descriptor story.

## Recommended Paper Architecture

### Introduction

1. OER is controlled by metal-oxygen bonding, but different catalyst families are usually rationalized with different descriptors.
2. Single-atom catalysts provide well-defined coordination environments, and O-coordinated sites on graphene/GO have been shown to be active for OER.
3. However, SAC studies usually focus on metal identity or coordination number, while oxide and metal surface studies focus on extended surface electronic structure.
4. This separation leaves an unresolved question: why can a one-atom model reproduce trends found on oxide/metal surfaces?
5. Here, we use single-atom, oxide-surface/metal-surface, and oxide-cluster calculations to show that local bonding angle controls metal-oxygen interaction and therefore OER activity.

## Results Storyline

### Result 1. Establish the single-atom OER reference model

**Question:** Can a one-atom OER model reproduce known activity trends?

**Suggested content:**

- Define your single-atom structure: metal center, O4 or related coordination, support model, charge/spin state if available.
- Calculate the standard OER pathway: `* -> *OH -> *O -> *OOH -> O2`.
- Report limiting potential/overpotential and potential-determining step.
- Compare the trend with known oxide surface or metal surface results.

**Main message:** The single-atom model already carries the essential OER bonding information.

**Figure 1:**

- Atomic structure of the single-atom site.
- OER free-energy diagram.
- Parity/trend plot comparing single-atom results with oxide/metal surface references.

### Result 2. Show that agreement with oxide/metal surfaces comes from local M-O bonding, not from global material type

**Question:** Why does the single atom agree with oxide and metal surfaces?

**Suggested content:**

- Compare key electronic descriptors: d-band center, p-band center, Bader charge, spin moment, ICOHP/COHP, projected DOS, metal-O covalency.
- Show that the OER intermediate adsorption energies scale with local M-O bonding metrics.
- Demonstrate that the same descriptor explains both single-atom and surface datasets.

**Main message:** The apparent cross-material agreement is caused by shared local metal-oxygen bonding motifs.

**Figure 2:**

- Projected DOS/COHP for representative single atom, oxide surface, and metal surface.
- Scaling between `ΔG_*O` or `ΔG_*OOH - ΔG_*OH` and local bonding descriptor.
- Color points by system type to show that they collapse onto one trend.

### Result 3. Introduce oxide clusters as a controllable bridge system

**Question:** Can oxide clusters reveal which local structural variable controls the bonding?

**Suggested content:**

- Construct oxide cluster models with systematically varied metal-oxygen-metal or oxygen-metal-oxygen angles.
- Keep metal identity and coordination as controlled as possible.
- Calculate OER intermediate adsorption and limiting potentials.
- Show that activity varies continuously with angle.

**Main message:** Oxide clusters expose the structural origin of the bonding descriptor: angle controls orbital overlap and M-O bond strength.

**Figure 3:**

- Oxide cluster structures sorted by angle.
- Plot: bond angle vs adsorption energies or overpotential.
- Representative charge density difference or orbital overlap diagrams.

### Result 4. Connect angle to bonding and OER activity

**Question:** What physically changes when the angle changes?

**Suggested content:**

- Use COHP/ICOHP, PDOS, charge density difference, crystal orbital overlap, or orbital-resolved projected DOS.
- Show how angle changes metal d and oxygen p overlap.
- Link this to stronger/weaker adsorption of `*OH`, `*O`, and `*OOH`.
- Identify whether the optimum follows Sabatier behavior: too weak and too strong bonding are both unfavorable.

**Main message:** Bond angle is a geometric handle for tuning electronic bonding, which determines OER activity.

**Figure 4:**

- Angle vs ICOHP or M-O covalency.
- ICOHP/PDOS vs OER overpotential.
- Schematic: acute/linear/open angle changes orbital overlap and intermediate stabilization.

### Result 5. Generalize the descriptor across single atoms, oxide clusters, oxide surfaces, and metal surfaces

**Question:** Is the descriptor transferable?

**Suggested content:**

- Combine all systems in one plot: single atom, oxide cluster, oxide surface, metal surface.
- Use the same x-axis: angle-derived bonding descriptor, direct angle, M-O bond order, or hybrid angle-bonding descriptor.
- Use the y-axis: overpotential, limiting potential, or key adsorption energy.
- Show whether a single volcano or monotonic trend describes all systems.

**Main message:** The local geometry-bonding descriptor unifies OER activity across material classes.

**Figure 5:**

- Universal map of OER activity.
- Classification of systems into regions: weak bonding, optimal bonding, strong bonding.
- Design rule for future catalysts.

## Recommended Abstract Logic

OER catalyst design is often separated into single-atom, metal-surface and oxide-surface frameworks, although all involve metal-oxygen bond formation and cleavage. Here we use first-principles calculations to show that a single-atom OER model reproduces the activity trends of oxide and metal surfaces because it preserves the local metal-oxygen bonding motif. By extending the analysis to oxide clusters, we identify the metal-oxygen bonding angle as a geometric variable that directly tunes orbital overlap, intermediate adsorption and OER limiting potentials. The resulting angle-bonding descriptor collapses single-atom, oxide-cluster and extended-surface data onto a common activity relationship. These results suggest that single-atom models can serve as minimal active-site representations for oxide OER catalysis and provide a geometry-based route for designing cluster and surface catalysts.

## Figure Plan

| Figure | Claim | Required evidence |
|---|---|---|
| Fig. 1 | Single-atom model gives credible OER activity | OER free energy, limiting step, comparison to known surfaces |
| Fig. 2 | Agreement with oxide/metal surfaces is local-bonding driven | PDOS/COHP/Bader/spin and adsorption scaling |
| Fig. 3 | Oxide clusters isolate angle effects | Cluster structures, angle series, adsorption energies |
| Fig. 4 | Angle changes bonding directly | Angle vs ICOHP/PDOS/charge-density difference |
| Fig. 5 | Descriptor generalizes across systems | Unified volcano/trend for SAC, cluster, oxide, metal |

## Claim-Evidence Map

| Claim | Evidence needed | Current status |
|---|---|---|
| One-atom OER model agrees with oxide/metal surfaces | Quantitative comparison of adsorption energies or overpotential trends | User says observed; needs plotted data |
| Local bonding, not material class, explains agreement | Common descriptor correlating SAC and surface results | Needs COHP/PDOS/Bader/spin analysis |
| Oxide cluster angle controls activity | Angle series with OER adsorption/free-energy changes | User says observed; needs systematic plot |
| Angle reflects bonding | Angle vs ICOHP/PDOS/charge-density difference | Needs bonding analysis |
| Descriptor is transferable | Unified plot across single atom, cluster, oxide surface, metal surface | Needs combined dataset |

## Key Controls to Avoid Reviewer Criticism

1. **Separate angle from bond length.** If angle changes also change M-O distance, reviewers may say activity is really bond length or coordination strength. Include partial correlation or 2D descriptor: angle + bond length.
2. **Check charge and spin.** For OER, metal oxidation state and spin can dominate adsorption. Report Bader charge/spin for the angle series.
3. **Use the same computational settings.** Same functional, U value if any, solvation correction, OER reference scheme and adsorbate corrections across SAC/cluster/surface.
4. **Show stability.** For clusters and single atoms, at least adsorption/formation energy or AIMD/phonon-style stability check if available.
5. **Clarify boundary.** Claim “local bonding unifies these selected OER systems”, not “all OER catalysts”.

## Best Story Version

The strongest version is a mechanism paper:

> Single-atom OER sites are minimal models of oxide active centers. Their apparent agreement with oxide and metal surfaces arises because OER is governed by local metal-oxygen bonding. Oxide clusters reveal that the key geometric control is the bonding angle, which tunes orbital overlap and therefore intermediate adsorption. This gives a transferable geometry-bonding descriptor for OER catalyst design.

## Revised Model Strategy: Vary the Local O-Atom Cluster around One Single Atom on Graphene

This is likely the most suitable framing for the current project.

The central model should not be a bare isolated metal atom, and it should also not become a conventional multi-metal nanoparticle. A stronger model is:

> one metal single atom anchored on graphene/graphene oxide, surrounded by a tunable local oxygen cluster or oxide-like coordination motif.

In this framing, the catalytic unit is the whole `M-Ox-G` motif, not only the `M` atom.

### Why this is better

1. It stays close to the Advanced Science 2020 `Ni-O-G` logic, where the active site is described as high-valence Ni coordinated to oxygen sites, and the computational model uses an oxygen-rich `Ni-O4(OH)2` environment.
2. It avoids an over-simplified single-atom story. OER involves oxygenated intermediates, so the surrounding O atoms are not passive ligands; they define oxidation state, orbital overlap and adsorbate binding.
3. It creates a natural bridge to oxide surfaces and oxide clusters. Oxide surfaces are extended `M-O` networks, while your graphene-supported `M-Ox` motif is a finite local analogue.
4. It gives you a controllable structural variable: the number, arrangement and angle of O atoms around the single atom.

### Recommended model series

Use one fixed metal center first, then vary only the surrounding oxygen motif:

- `M-C4/G` or weakly O-containing graphene: carbon-dominated reference.
- `M-O1C3/G`: one local O ligand.
- `M-O2C2/G`: mixed O/C coordination.
- `M-O3C1/G`: oxygen-rich coordination.
- `M-O4/G`: square-planar or distorted O4 coordination.
- `M-O4(OH)x/G`: OER-relevant hydroxylated high-valence state, similar in spirit to the `Ni-O4(OH)2` model.
- `M-Ox-cluster/G`: a finite oxide-like motif where neighboring O atoms form different `O-M-O` or `M-O-M` angles.

After this, extend to several metals only if the geometric rule is already clear.

### What to avoid

- Avoid calling it a single-atom catalyst if the active center becomes a multi-metal cluster.
- Avoid changing metal identity, O coordination number, bond length and support defect all at once. Reviewers will not know which variable causes the activity change.
- Avoid making the story only about `O4` as a static coordination number. The stronger point is that the local oxygen motif can deform and tune bonding angles.

### New central claim

> The OER activity of graphene-supported single-atom catalysts is governed by the geometry and bonding of a single-atom-centered oxygen cluster, rather than by the isolated metal atom alone.

### Revised figure logic

| Figure | Revised claim |
|---|---|
| Fig. 1 | Build a graphene-supported `M-Ox` model series and show that oxygen-rich coordination changes OER activity. |
| Fig. 2 | Show that `M-O4` or hydroxylated `M-O4(OH)x` reproduces oxide/metal-surface-like OER trends. |
| Fig. 3 | Vary the local O-cluster angle around one metal atom and show activity changes. |
| Fig. 4 | Explain the angle effect using M-O bonding, PDOS, COHP/ICOHP, charge and spin. |
| Fig. 5 | Generalize the `M-Ox` motif as a local descriptor connecting graphene SACs, oxide clusters and oxide surfaces. |

## Weaker Version if Data Are Limited

If the surface comparison is only qualitative, frame it as:

> We propose and test a local-geometry explanation for why single-atom OER models reproduce selected surface-like trends.

Avoid claiming a universal descriptor unless the cross-system correlation is strong.
