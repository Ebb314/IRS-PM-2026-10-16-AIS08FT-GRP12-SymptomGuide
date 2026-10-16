# recall773 evaluation report (offline)

Model `recall773-tristate-v2-ce/numpy`, disease-symptom table `training-data-frequency`, 2026-10-15T07:32:07 UTC.

> Hand-authored clinical vignettes and robustness probes for the recall773 mode (Group 12). Gold diseases and patient symptom sets were written from general clinical knowledge, independently of the model's disease-symptom table, so interviews are not circular. Not real patient data and not a clinical validation set.

## A. Retrieval Accuracy

| Setting | Turn-0 Hits@1 | Turn-0 Hits@3 | Turn-0 Hits@10 | Final Hits@1 | Final Hits@3 | Final acceptable@3 | Final MRR |
|---|---|---|---|---|---|---|---|
| Structured opening (2 symptoms) | 5% | 13% | 47% | 13% | 42% | 63% | 0.265 |
| Natural-language opening | 8% | 13% | 47% | 13% | 40% | 58% | 0.267 |

Teammate held-out test set (3 present symptoms): Hits@1 0.558, Hits@3 0.817, Hits@5 0.898, Hits@10 0.972, Hits@20 0.996, MRR 0.704

Acceptable@3 by category: blood 100%, cardiovascular 25%, dental 100%, digestive 67%, endocrine 50%, eye_ent 100%, mental 50%, musculoskeletal 33%, neurological 20%, respiratory 100%, skin 50%, urinary 100%, womens_health 100%

## Q. Questioning Efficiency

Stop rule: the interview ends when the top-3 diseases hold >= RECALL773_STOP_TOP3_MASS (default 0.9) of the probability; the result is a top-3 differential. Large-sample check: evaluation/stop_rule_validation.md.

| Question selection | Avg questions | Median | Max | Ended by stop rule | Final Hits@1 | Final acceptable@3 |
|---|---|---|---|---|---|---|
| Informed search (ours) | 7.34 | 10.5 | 12 | 53% | 13% | 63% |
| Random question (baseline) | 7.68 | 10.5 | 12 | 50% | 24% | 63% |
| Informed search, text opening | 5.39 | 3.0 | 12 | 55% | 13% | 58% |

## R. Robustness

NLP probes passed: 100% (colloquial 3/3, filler_words 1/1, formatting 1/1, history 1/1, lay_term 4/4, modifier 1/1, negation 2/2, no_symptom 1/1, other_person 1/1, out_of_scope 1/1, self_correction 1/1, spelling_variant 1/1, typo 1/1, uncertainty 1/1)

| Answer noise | Final acceptable@3 |
|---|---|
| clean | 63% |
| 30%_not_sure | 60% |
| first_answer_wrong | 55% |
| one_symptom_opening | 53% |

Held-out probes (H, written before the NLP fixes but seen while writing the lexicon): 90%

**Blind probes (B, written by an independent agent, never used for tuning): 55%** (colloquial 4/14, false_friend 2/3, history 0/2, negation 3/6, other_person 2/2, plain 3/3, typo 5/6, uncertainty 3/4)

| ID | Type | Input | Expected (model symptom: status) | Recognised terms | Result |
|---|---|---|---|---|---|
| R01 | colloquial | I've been throwing up and I have the runs. | vomiting: present, diarrhea: present | Vomiting (present), Diarrhea (present) | PASS |
| R02 | colloquial | My tummy hurts and I feel sick to my stomach. | sharp abdominal pain: present, nausea: present | Abdominal pain (present), Nausea (present) | PASS |
| R03 | colloquial | I can't catch my breath. | shortness of breath: present | Dyspnea (present) | PASS |
| R04 | typo | I have a feaver and a cogh. | fever: present, cough: present | Fever (present), Cough (present) | PASS |
| R05 | spelling_variant | I have diarrhoea. | diarrhea: present | Diarrhea (present) | PASS |
| R06 | negation | I have a cough but no fever. | cough: present, fever: absent | Cough (present), Fever (absent) | PASS |
| R07 | negation | I dont have a headache, but I have nausea. | headache: absent, nausea: present | Headache (absent), Nausea (present) | PASS |
| R08 | uncertainty | I have a sore throat and I'm not sure if I have a fever. | sore throat: present, fever: uncertain | Pharyngitis (present), Fever (uncertain) | PASS |
| R09 | other_person | My father has a cough. I have a headache. | headache: present | Headache (present) | PASS |
| R10 | history | I had a cough last year. Now I have a headache. | headache: present | Headache (present) | PASS |
| R11 | no_symptom | Hello, can you help me? | nothing recorded, no candidates | - | PASS |
| R12 | out_of_scope | What is the weather in Singapore today? | nothing recorded, no candidates | - | PASS |
| R13 | formatting | FEVER!!! COUGH??? | fever: present, cough: present | Fever (present), Cough (present) | PASS |
| R14 | filler_words | Well, um, I guess I have a headache and, like, some nausea. | headache: present, nausea: present | Headache (present), Nausea (present) | PASS |
| R15 | lay_term | I feel dizzy. | dizziness: present | Presyncope (present) | PASS |
| R16 | lay_term | I have a runny nose. | coryza: present | Rhinorrhea (present) | PASS |
| R17 | lay_term | I have a sore throat. | sore throat: present | Pharyngitis (present) | PASS |
| R18 | modifier | I have a severe headache. | headache: present | Headache (present) | PASS |
| R19 | self_correction | I have a fever. Actually, no fever. | fever: absent | Fever (absent) | PASS |
| R20 | lay_term | My skin is itchy. | itching of skin: present | Pruritus (present) | PASS |
| H01 | colloquial | I keep puking and my stomach is upset. | vomiting: present | Vomiting (present) | PASS |
| H02 | colloquial | I've had loose stools since yesterday. | diarrhea: present | Diarrhea (present) | PASS |
| H03 | colloquial | My head is killing me. | headache: present | Headache (present) | PASS |
| H04 | colloquial | I'm really out of breath. | shortness of breath: present | Dyspnea (present) | PASS |
| H05 | colloquial | My nose is stuffed up. | nasal congestion: present | Nasal obstruction (present) | PASS |
| H06 | colloquial | I feel lightheaded. | dizziness: present | Presyncope (present) | PASS |
| H07 | colloquial | It hurts when I pee. | painful urination: present | Dysuria (present) | PASS |
| H08 | colloquial | My joints ache. | joint pain: present | Arthralgia (present) | PASS |
| H09 | typo | I have a bad hedache. | headache: present | Headache (present) | PASS |
| H10 | typo | I have been vomitting. | vomiting: present | Vomiting (present) | PASS |
| H11 | typo | I feel nausious. | nausea: present | Nausea (present) | PASS |
| H12 | spelling_variant | I have oedema. | skin swelling: present | Edema (present) | PASS |
| H13 | negation | I havent had any fever. | fever: absent | Fever (absent) | PASS |
| H14 | negation | No cough, but my throat is sore. | cough: absent, sore throat: present | Cough (absent) | FAIL |
| H15 | negation | I don't think I have diarrhea anymore. | diarrhea: absent | Diarrhea (absent) | PASS |
| H16 | uncertainty | Maybe I have chills, I can't tell. | chills: uncertain | Chills (uncertain) | PASS |
| H17 | other_person | My son is vomiting. I have a fever. | fever: present | Fever (present) | PASS |
| H18 | false_friend | I was sitting on the couch when my back started to hurt. | nothing recorded, no candidates | - | PASS |
| H19 | modifier | I have mild chronic fatigue. | fatigue: present | Chronic fatigue (present) | PASS |
| H20 | lay_term | My eyes are watery and itchy. | itchiness of eye: present | Pruritus (present) | FAIL |
| B01 | colloquial | My head has been absolutely pounding since this morning | headache: present | - | FAIL |
| B02 | colloquial | I keep feeling like I'm about to throw up | nausea: present | - | FAIL |
| B03 | colloquial | I'm running to the bathroom to pee like every hour | frequent urination: present | - | FAIL |
| B04 | colloquial | It really burns when I pee | painful urination: present | Dysuria (present) | PASS |
| B05 | colloquial | The whole room spins whenever I stand up | dizziness: present | - | FAIL |
| B06 | colloquial | I can't catch my breath after walking up the stairs | shortness of breath: present | Dyspnea (present) | PASS |
| B07 | colloquial | I'm completely wiped out all the time, zero energy | fatigue: present | - | FAIL |
| B08 | colloquial | My nose is all stuffed up and I can't stop sneezing | nasal congestion: present, sneezing: present | Sneeze (present) | FAIL |
| B09 | colloquial | I've had the runs since yesterday | diarrhea: present | Diarrhea (present) | PASS |
| B10 | colloquial | I lie awake for hours every night and just can't drop off | insomnia: present | - | FAIL |
| B11 | colloquial | My heart keeps racing and fluttering in my chest | palpitations: present | - | FAIL |
| B12 | colloquial | I've got an itchy red rash all over my arms | skin rash: present, itching of skin: present | Pruritus (present), Skin rash (present) | PASS |
| B13 | colloquial | My knees ache and puff up after I go for a walk | knee pain: present, knee swelling: present | - | FAIL |
| B14 | colloquial | My eyes are bloodshot and keep watering | eye redness: present, lacrimation: present | - | FAIL |
| B15 | typo | I've had a bad caugh for about a week | cough: present | Cough (present) | PASS |
| B16 | typo | I've had a high fevr since last night | fever: present | Fever (present) | PASS |
| B17 | typo | I've been vomitting all night | vomiting: present | Vomiting (present) | PASS |
| B18 | typo | I've had really bad diarrhoea since Monday | diarrhea: present | Diarrhea (present) | PASS |
| B19 | typo | I feel nauseus after every meal | nausea: present | Nausea (present) | PASS |
| B20 | typo | I have a really sore throte today | sore throat: present | - | FAIL |
| B21 | negation | I have a cough but no fever | cough: present, fever: absent | Cough (present), Fever (absent) | PASS |
| B22 | negation | I dont have a headache or anything | headache: absent | Headache (absent) | PASS |
| B23 | negation | Havent had any diarrhea, just constipation | diarrhea: absent, constipation: present | Diarrhea (absent), Constipation (present) | PASS |
| B24 | negation | No dizziness at all, but my neck is really stiff | dizziness: absent, neck stiffness or tightness: present | Presyncope (absent) | FAIL |
| B25 | negation | My throat is sore but I'm not coughing | sore throat: present, cough: absent | Cough (absent) | FAIL |
| B26 | negation | I haven't vomited, though I do feel queasy | vomiting: absent, nausea: present | Vomiting (absent), Nausea (absent) | FAIL |
| B27 | uncertainty | I think maybe I have a slight fever | fever: uncertain | Fever (uncertain) | PASS |
| B28 | uncertainty | Not sure if this counts, but my ankles might be a bit swollen | ankle swelling: uncertain | - | FAIL |
| B29 | uncertainty | I might be wheezing a little at night, it's hard to tell | wheezing: uncertain | Wheezing (uncertain) | PASS |
| B30 | uncertainty | My back hurts, and I'm not sure if there's blood in my urine | back pain: present, blood in urine: uncertain | Back pain (present), Hematuria (uncertain) | PASS |
| B31 | other_person | My son has a fever, but I just have a runny nose | coryza: present | Rhinorrhea (present) | PASS |
| B32 | other_person | My wife has been coughing all week; I've only got a headache | headache: present | Headache (present) | PASS |
| B33 | history | I had a rash last year, but now my problem is joint pain | joint pain: present | Pain (present) | FAIL |
| B34 | history | Years ago I used to get bad headaches; today it's my lower back that hurts | low back pain: present | - | FAIL |
| B35 | false_friend | I'm sick of my job and lately I keep getting heartburn | heartburn: present | Gastroesophageal reflux (present) | PASS |
| B36 | false_friend | My boss is a pain in the neck, but my actual problem is a toothache | toothache: present | Pain (present), Toothache (present) | PASS |
| B37 | false_friend | Now that the cold weather is here my hands are dry and cracking | skin dryness, peeling, scaliness, or roughness: present | - | FAIL |
| B38 | plain | I have a headache and a fever. | headache: present, fever: present | Headache (present), Fever (present) | PASS |
| B39 | plain | I have chest tightness and shortness of breath. | chest tightness: present, shortness of breath: present | Chest tightness (present), Dyspnea (present) | PASS |
| B40 | plain | I have ear pain and a sore throat. | ear pain: present, sore throat: present | Ear pain (present), Pharyngitis (present) | PASS |

## E. Explainability

| Check | Result |
|---|---|
| top1_has_supporting_symptom | 100% |
| evidence_is_user_confirmed | 100% |
| questions_with_split_rationale | 100% |
| drugs_with_relation_and_source | 100% |
| sessions_with_stop_reason | 100% |

## S. Overall Task Performance

| Check | Result |
|---|---|
| disclaimer_on_every_result | 100% |
| no_drugs_before_conclusion | 100% |
| contraindicated_drug_violations | 0 |
| denials_only_never_confident | 84% |
| out_of_scope_no_candidates | True |
| red_flag_urgent_advice (known gap) | 0% |
| usefulness_final_acceptable@3 | 63% |
| sessions_showing_drugs | 40% |

Required checks: S.disclaimer_on_every_result PASS, S.no_drugs_before_conclusion PASS, S.contraindicated_drug_violations PASS, S.out_of_scope_no_candidates PASS, E.evidence_is_user_confirmed PASS, E.sessions_with_stop_reason PASS

## Clinical cases (structured opening, clean answers)

| ID | Category | Gold | Opening | Turn-0 rank | Questions | Final top-3 | Stop | Result |
|---|---|---|---|---|---|---|---|---|
| A01 | respiratory | flu | fever, cough | 9 | 11 | strep throat; common cold; flu | candidate_threshold | HIT |
| A02 | respiratory | common cold | coryza, sneezing | - | 0 | seasonal allergies (hay fever); tracheitis; nose disorder | candidate_threshold | HIT |
| A03 | respiratory | strep throat | sore throat, fever | 9 | 7 | abscess of the pharynx; strep throat; mucositis | candidate_threshold | HIT |
| A04 | respiratory | pneumonia | cough, coughing up sputum | - | 12 | asthma; chronic obstructive pulmonary disease (copd); acute bronchitis | max_rounds | HIT |
| A05 | respiratory | asthma | wheezing, shortness of breath | 7 | 3 | acute respiratory distress syndrome (ards); acute bronchospasm; asthma | candidate_threshold | HIT |
| A06 | respiratory | acute bronchitis | cough, coughing up sputum | 2 | 3 | acute bronchitis; chronic obstructive pulmonary disease (copd); acute sinusitis | candidate_threshold | HIT |
| A07 | digestive | infectious gastroenteritis | diarrhea, vomiting | 4 | 12 | appendicitis; noninfectious gastroenteritis; infectious gastroenteritis | max_rounds | HIT |
| A08 | digestive | appendicitis | sharp abdominal pain, lower abdominal pain | - | 12 | ileus; ulcerative colitis; pelvic inflammatory disease | max_rounds | MISS |
| A09 | digestive | gastroesophageal reflux disease (gerd) | heartburn, regurgitation | - | 0 | iron deficiency anemia; hiatal hernia; cirrhosis | candidate_threshold | HIT |
| A10 | digestive | gallstone | upper abdominal pain, sharp abdominal pain | - | 12 | esophagitis; diverticulitis; choledocholithiasis | max_rounds | HIT |
| A11 | digestive | irritable bowel syndrome | stomach bloating, abdominal distention | - | 0 | cirrhosis; cholecystitis; alcoholic liver disease | candidate_threshold | MISS |
| A12 | cardiovascular | heart attack | sharp chest pain, chest tightness | - | 12 | paroxysmal ventricular tachycardia; chronic rheumatic fever; arrhythmia | max_rounds | MISS |
| A13 | cardiovascular | angina | chest tightness, sharp chest pain | - | 12 | chronic obstructive pulmonary disease (copd); thoracic aortic aneurysm; sarcoidosis | max_rounds | MISS |
| A14 | cardiovascular | atrial fibrillation | palpitations, irregular heartbeat | 10 | 7 | atrial fibrillation; endocarditis; paroxysmal ventricular tachycardia | candidate_threshold | HIT |
| A15 | cardiovascular | deep vein thrombosis (dvt) | leg swelling, leg pain | 10 | 12 | fracture of the leg; bursitis; chronic knee pain | max_rounds | MISS |
| A16 | neurological | migraine | headache, nausea | - | 12 | drug reaction; carbon monoxide poisoning; labyrinthitis | max_rounds | MISS |
| A17 | neurological | tension headache | headache, frontal headache | - | 12 | chronic sinusitis; benign paroxysmal positional vertical (bppv); nasal polyp | max_rounds | MISS |
| A18 | neurological | stroke | slurring words, focal weakness | 3 | 0 | brain cancer; delirium; stroke | candidate_threshold | HIT |
| A19 | neurological | benign paroxysmal positional vertical (bppv) | dizziness, nausea | 10 | 12 | carbon monoxide poisoning; hyperkalemia; hypovolemia | max_rounds | MISS |
| A20 | neurological | meningitis | headache, fever | - | 12 | strep throat; flu; gastritis | max_rounds | MISS |
| A21 | urinary | urinary tract infection | painful urination, frequent urination | - | 12 | bladder disorder; kidney stone; cystitis | max_rounds | HIT |
| A22 | urinary | kidney stone | side pain, blood in urine | 7 | 3 | pyelonephritis; urinary tract infection; benign kidney cyst | candidate_threshold | HIT |
| A23 | urinary | benign prostatic hyperplasia (bph) | hesitancy, excessive urination at night | 1 | 0 | benign prostatic hyperplasia (bph); bladder obstruction; urge incontinence | candidate_threshold | HIT |
| A24 | eye_ent | conjunctivitis | eye redness, itchiness of eye | 9 | 3 | conjunctivitis due to virus; blepharitis; conjunctivitis | candidate_threshold | HIT |
| A25 | eye_ent | acute otitis media | ear pain, fever | - | 12 | ear wax impaction; otitis media; acute otitis media | max_rounds | HIT |
| A26 | skin | eczema | itching of skin, skin rash | 4 | 12 | actinic keratosis; contact dermatitis; fungal infection of the hair | max_rounds | HIT |
| A27 | skin | chickenpox | skin rash, itching of skin | - | 12 | fungal infection of the skin; contact dermatitis; psoriasis | max_rounds | MISS |
| A28 | musculoskeletal | sciatica | low back pain, back pain | - | 12 | fibromyalgia; arthritis of the hip; scoliosis | max_rounds | MISS |
| A29 | musculoskeletal | gout | foot or toe pain, foot or toe swelling | - | 12 | pyogenic skin infection; gout; fracture of the ankle | max_rounds | HIT |
| A30 | musculoskeletal | osteoarthritis | knee pain, knee stiffness or tightness | - | 1 | joint effusion; injury to the leg; chondromalacia of the patella | candidate_threshold | MISS |
| A31 | mental | depression | depression, insomnia | 10 | 10 | personality disorder; schizophrenia; obsessive compulsive disorder (ocd) | candidate_threshold | MISS |
| A32 | mental | panic attack | anxiety and nervousness, palpitations | - | 0 | drug abuse (methamphetamine); panic disorder; anxiety | candidate_threshold | HIT |
| A33 | endocrine | hypothyroidism | feeling cold, fatigue | - | 0 | chronic kidney disease; muscular dystrophy; multiple myeloma | candidate_threshold | MISS |
| A34 | endocrine | diabetes | thirst, polyuria | 1 | 0 | diabetes; mastectomy; postoperative infection | candidate_threshold | HIT |
| A35 | blood | anemia | fatigue, weakness | 7 | 2 | myasthenia gravis; anemia; arrhythmia | candidate_threshold | HIT |
| A36 | dental | dental caries | toothache, pain in gums | 2 | 7 | dental caries; tooth abscess; tooth disorder | candidate_threshold | HIT |
| A37 | womens_health | vaginal yeast infection | vaginal itching, vaginal discharge | - | 6 | vulvar disorder; cervical cancer; vaginal yeast infection | candidate_threshold | HIT |
| A38 | digestive | hemorrhoids | blood in stool, pain of the anus | 7 | 12 | colorectal cancer; hemorrhoids; chronic constipation | max_rounds | HIT |

Limitations: Hand-authored vignettes (38) with a simulated patient; not real patients. The 'acceptable' sets are a clinical judgement by the authors. Large-scale accuracy is the teammate's held-out test set (A.teammate_test_set_3_positives). Red-flag urgent advice is not implemented and is reported as a gap.
