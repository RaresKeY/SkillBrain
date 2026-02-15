# Automated Spam/Ham Comparison Report

Generated (UTC): 2026-02-15 19:51:01

## Scope
- Real dataset: `/home/mintmainog/workspace/vs_code_workspace/SkillBrain_Python_Homework_Fork/assignments/hw_adv/a_35_0_spam_ham/spam.csv` (5574 rows)
- Custom dataset: `/home/mintmainog/workspace/vs_code_workspace/SkillBrain_Python_Homework_Fork/assignments/hw_adv/a_35_0_spam_ham/custom_test_dataset.csv` (24 rows)
- Compared methods:
  - TF-IDF + MultinomialNB
  - Static embeddings v2 + LogisticRegression

## Model Data
- Static embeddings repo: `LogicLark-QuantumQuill/static-embeddings-en-50m-v2`
- Loaded embedding file: `/home/mintmainog/workspace/vs_code_workspace/SkillBrain_Python_Homework_Fork/assignments/hw_adv/a_35_0_spam_ham/models/LogicLark-QuantumQuill__static-embeddings-en-50m-v2/static_embeddings_en_50m_pruned_fp16_v2.safetensors`
- Static train token coverage: 89.86%
- Static holdout token coverage: 89.81%
- Static custom token coverage: 98.25%

## Holdout Test (From Real Dataset)
| Method | Accuracy | Precision (spam) | Recall (spam) | F1 (spam) |
|---|---:|---:|---:|---:|
| TF-IDF + NB | 97.49% | 100.00% | 81.21% | 89.63% |
| Static-v2 + LogReg | 95.87% | 85.52% | 83.22% | 84.35% |

Winner (F1 spam): **TF-IDF + NB**

### Confusion Matrix: TF-IDF + NB
|   | Pred HAM | Pred SPAM |
|---|---:|---:|
| True HAM | 966 | 0 |
| True SPAM | 28 | 121 |

### Confusion Matrix: Static-v2 + LogReg
|   | Pred HAM | Pred SPAM |
|---|---:|---:|
| True HAM | 945 | 21 |
| True SPAM | 25 | 124 |

## Custom Dataset Test
| Method | Accuracy | Precision (spam) | Recall (spam) | F1 (spam) |
|---|---:|---:|---:|---:|
| TF-IDF + NB | 70.83% | 100.00% | 41.67% | 58.82% |
| Static-v2 + LogReg | 87.50% | 90.91% | 83.33% | 86.96% |

Winner (F1 spam): **Static-v2 + LogReg**

### Confusion Matrix: TF-IDF + NB
|   | Pred HAM | Pred SPAM |
|---|---:|---:|
| True HAM | 12 | 0 |
| True SPAM | 7 | 5 |

### Confusion Matrix: Static-v2 + LogReg
|   | Pred HAM | Pred SPAM |
|---|---:|---:|
| True HAM | 11 | 1 |
| True SPAM | 2 | 10 |

## Color-Coded Example Predictions (Custom Dataset)
Legend: HAM/green, SPAM/red, PASS=correct, FAIL=wrong.

| # | Text | True | TF-IDF+NB | Static-v2+LogReg |
|---:|---|---|---|---|
| 1 | hey are we still meeting at 6 near the station? | <span style="color:#166534;font-weight:700;">HAM</span> | <span style="color:#166534;font-weight:700;">HAM</span> <span style="color:#166534;font-weight:700;">PASS</span> | <span style="color:#166534;font-weight:700;">HAM</span> <span style="color:#166534;font-weight:700;">PASS</span> |
| 2 | can you send me the grocery list before i leave work? | <span style="color:#166534;font-weight:700;">HAM</span> | <span style="color:#166534;font-weight:700;">HAM</span> <span style="color:#166534;font-weight:700;">PASS</span> | <span style="color:#166534;font-weight:700;">HAM</span> <span style="color:#166534;font-weight:700;">PASS</span> |
| 3 | thanks for the notes yesterday they helped a lot. | <span style="color:#166534;font-weight:700;">HAM</span> | <span style="color:#166534;font-weight:700;">HAM</span> <span style="color:#166534;font-weight:700;">PASS</span> | <span style="color:#166534;font-weight:700;">HAM</span> <span style="color:#166534;font-weight:700;">PASS</span> |
| 4 | i'll call you after dinner when i get home. | <span style="color:#166534;font-weight:700;">HAM</span> | <span style="color:#166534;font-weight:700;">HAM</span> <span style="color:#166534;font-weight:700;">PASS</span> | <span style="color:#166534;font-weight:700;">HAM</span> <span style="color:#166534;font-weight:700;">PASS</span> |
| 5 | reminder your dentist appointment is tomorrow at 10am. | <span style="color:#166534;font-weight:700;">HAM</span> | <span style="color:#166534;font-weight:700;">HAM</span> <span style="color:#166534;font-weight:700;">PASS</span> | <span style="color:#166534;font-weight:700;">HAM</span> <span style="color:#166534;font-weight:700;">PASS</span> |
| 6 | the package arrived safe and i left it at reception. | <span style="color:#166534;font-weight:700;">HAM</span> | <span style="color:#166534;font-weight:700;">HAM</span> <span style="color:#166534;font-weight:700;">PASS</span> | <span style="color:#166534;font-weight:700;">HAM</span> <span style="color:#166534;font-weight:700;">PASS</span> |
| 7 | please review the draft and share comments by friday. | <span style="color:#166534;font-weight:700;">HAM</span> | <span style="color:#166534;font-weight:700;">HAM</span> <span style="color:#166534;font-weight:700;">PASS</span> | <span style="color:#b91c1c;font-weight:700;">SPAM</span> <span style="color:#b91c1c;font-weight:700;">FAIL</span> |
| 8 | movie night at my place bring snacks if you can. | <span style="color:#166534;font-weight:700;">HAM</span> | <span style="color:#166534;font-weight:700;">HAM</span> <span style="color:#166534;font-weight:700;">PASS</span> | <span style="color:#166534;font-weight:700;">HAM</span> <span style="color:#166534;font-weight:700;">PASS</span> |
| 9 | your one time password for sign in is 482193. | <span style="color:#166534;font-weight:700;">HAM</span> | <span style="color:#166534;font-weight:700;">HAM</span> <span style="color:#166534;font-weight:700;">PASS</span> | <span style="color:#166534;font-weight:700;">HAM</span> <span style="color:#166534;font-weight:700;">PASS</span> |
| 10 | train delayed by 15 minutes i will be late. | <span style="color:#166534;font-weight:700;">HAM</span> | <span style="color:#166534;font-weight:700;">HAM</span> <span style="color:#166534;font-weight:700;">PASS</span> | <span style="color:#166534;font-weight:700;">HAM</span> <span style="color:#166534;font-weight:700;">PASS</span> |
| 11 | happy birthday hope you have a great day. | <span style="color:#166534;font-weight:700;">HAM</span> | <span style="color:#166534;font-weight:700;">HAM</span> <span style="color:#166534;font-weight:700;">PASS</span> | <span style="color:#166534;font-weight:700;">HAM</span> <span style="color:#166534;font-weight:700;">PASS</span> |
| 12 | can we reschedule the meeting to next tuesday? | <span style="color:#166534;font-weight:700;">HAM</span> | <span style="color:#166534;font-weight:700;">HAM</span> <span style="color:#166534;font-weight:700;">PASS</span> | <span style="color:#166534;font-weight:700;">HAM</span> <span style="color:#166534;font-weight:700;">PASS</span> |
| 13 | urgent you have won a 1000 dollar voucher claim now. | <span style="color:#b91c1c;font-weight:700;">SPAM</span> | <span style="color:#b91c1c;font-weight:700;">SPAM</span> <span style="color:#166534;font-weight:700;">PASS</span> | <span style="color:#b91c1c;font-weight:700;">SPAM</span> <span style="color:#166534;font-weight:700;">PASS</span> |
| 14 | free entry in 2 a weekly draw text win to 80085 now. | <span style="color:#b91c1c;font-weight:700;">SPAM</span> | <span style="color:#b91c1c;font-weight:700;">SPAM</span> <span style="color:#166534;font-weight:700;">PASS</span> | <span style="color:#b91c1c;font-weight:700;">SPAM</span> <span style="color:#166534;font-weight:700;">PASS</span> |
| 15 | congratulations selected for cash reward click the link immediately. | <span style="color:#b91c1c;font-weight:700;">SPAM</span> | <span style="color:#b91c1c;font-weight:700;">SPAM</span> <span style="color:#166534;font-weight:700;">PASS</span> | <span style="color:#b91c1c;font-weight:700;">SPAM</span> <span style="color:#166534;font-weight:700;">PASS</span> |
| 16 | limited offer buy now and get 70 percent off today only. | <span style="color:#b91c1c;font-weight:700;">SPAM</span> | <span style="color:#166534;font-weight:700;">HAM</span> <span style="color:#b91c1c;font-weight:700;">FAIL</span> | <span style="color:#b91c1c;font-weight:700;">SPAM</span> <span style="color:#166534;font-weight:700;">PASS</span> |
