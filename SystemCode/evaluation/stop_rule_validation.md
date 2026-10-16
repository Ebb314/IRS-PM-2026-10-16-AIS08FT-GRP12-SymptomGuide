# Stop rule validation (500 held-out test records)

Split rebuilt and verified against the bundle audit {'train': 151899, 'val': 18774, 'test': 18974}; dataset sha256 `8de90603ebada467...`. 256 diseases in the sample, 5.41 present symptoms per record.

Opening turn (2 symptoms): Hits@1 32%, Hits@3 59%, Hits@10 88%

| Rule | Stopped by rule | Avg questions | Hits@1 | Hits@3 | Hits@3 when rule fired | Hits@3 at 12 rounds | All-No stopped by rule |
|---|---|---|---|---|---|---|---|
| current: top1>=0.80 and top1+top2>=0.95 | 62% | 7.49 | 71% | 87% | 99% | 67% | 14% |
| top-3 mass >= 0.60 | 92% | 2.78 | 39% | 75% | 81% | 15% | 61% |
| top-3 mass >= 0.70 | 88% | 3.55 | 42% | 80% | 88% | 25% | 53% |
| top-3 mass >= 0.75 | 86% | 3.95 | 44% | 82% | 91% | 28% | 50% |
| top-3 mass >= 0.80 | 85% | 4.38 | 46% | 84% | 93% | 29% | 44% |
| top-3 mass >= 0.85 | 83% | 4.73 | 47% | 86% | 97% | 37% | 39% |
| top-3 mass >= 0.90 | 81% | 5.01 | 50% | 87% | 99% | 36% | 35% |

Implemented rule end to end (top-3 mass >= 0.9): stopped by rule 78%, 5.04 questions on average, Hits@1 50%, Hits@3 85% (99% when the rule fired); stop reasons {'candidate_threshold': 389, 'max_rounds': 108, 'no_questions': 3}.

Regression floors: hits@3 >= 0.8: PASS, avg_questions <= 6.0: PASS, hits@3_when_rule_fired >= 0.95: PASS

Held-out TEST records of the synthetic Disease-Symptom Dataset; the simulated patient answers from the record (1 = Yes, 0 = No, the teammate's synthetic-negative assumption). Gold is the exact record label (no acceptable alternatives). In-distribution, so expect higher accuracy than the hand-written clinical vignettes.
