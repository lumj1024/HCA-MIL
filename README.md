HCA-MIL: Hierarchical Cross-Granularity Attention Enhanced Multiple Instance Learning for Leukemia Subtyping
Leukemia subtyping from bone marrow cell images requires
fine-grained discrimination of subtle and heterogeneous cellular patterns,
while diagnostic evidence is often distributed across multiple cellular instances. Instead of treating each cell independently, we formulate subtype assessment as patient-level inference from collective cellular evidence under a weakly supervised multiple instance learning (MIL) setting. Existing MIL approaches mainly rely on direct instance embedding aggregation, leaving fine-grained intra-cell morphology and intercell population evidence insufficiently modeled. We present HCA-MIL, a
hierarchical cross-granularity attentive MIL framework that refines cellular representations prior to bag aggregation. With a GroupMambabased instance encoder, we introduce two complementary branches. The
intra-cell branch enhances diagnostically relevant subcellular structures
through sparse token routing and cross-group channel mixing, while the
inter-cell fusion branch couples refined cellular morphology with global
bag semantics. In this way, we obtain patient-level representations that
preserve both local morphological specificity and population-level evidence. On a clinical cohort of bone marrow smears, HCA-MIL consistently surpasses representative MIL baselines, achieving 95.93% accuracy, 99.35% AUC, and 95.89% F1-score across three random seeds.
Evaluation on Camelyon+ further supports the transferability of our
cross-granularity aggregation design. Our code is available at https:
//github.com/lumj1024/HCA-MIL
