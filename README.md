Domain: Healthcare

# PharmEasy Healthcare Decision Intelligence

Capstone project for the Minor in AI & Data Science (IIT Mandi). The data is synthetic and comes entirely from `generate_data.py`; nothing here uses or describes real PharmEasy records, and nothing here is clinical advice. Everything is computed with plain Python and NumPy.

Repository layout:

- `generate_data.py` - the provided generator, unchanged
- `data/records.csv` - generator output (800 rows)
- `common.py` - loading, the 560/240 split, standardisation, logistic regression, KNN and metrics, shared by Parts 2 and 4
- `part1.py` ... `part4.py` - one script per Part
- `outputs/` - every results file and CSV

## How to run

From a clean clone, in the repository root:

```bash
pip install -r requirements.txt
python generate_data.py
python part1.py
python part2.py
python part3.py
python part4.py
```

`generate_data.py` should print `800 10 170 e26b46e82f1a1af44dcabc133050802e`. Each part script overwrites its own files in `outputs/`. Part 4 refits the Part 2 models through `common.py`, so it does not depend on Part 2 having been run first.

## Part 1 findings

**R1.1** H0 (mean care-plan cost = 8.0, i.e. INR 8,000) is rejected: `z_stat = 2.8930`, which is above the critical value 1.96. A Type I error here would mean telling the finance team the INR 8,000 planning figure is wrong when the true average really is INR 8,000. The gap is `mean_value - 8.0 = 8.9382 - 8.0 = 0.9382`, so about INR 938 per patient, roughly 12% above plan. I think that matters in practice: across a few thousand care plans that is a budget shortfall large enough to notice, so the planning figure should be revised up rather than just flagged as "significant".

**R1.2** `n_required = 1293`, but only 800 rows are available. With 800 rows the 95% margin is 1.96 × 0.3243 ≈ 0.64 (about INR 636), not the INR 500 the team asked for. The current data is enough to say the average is above INR 8,000, but to pin it down to ±INR 500 the team needs roughly 500 more records.

**R1.3** `gd_slope_gap = 0.0000`. The cost J is a convex bowl in (m, c) with a single minimum, and with standardised X4 and η = 0.1 every step moves straight down that bowl, so after 500 updates gradient descent sits on the same point the closed form solves for directly (J stopped changing after about 100 updates: `gd_J100 = gd_J500 = 40.3698`).

**R1.4** `resid_sd_ratio_raw = 3.7525`: residuals in the upper half of fitted values are almost four times as spread out as in the lower half, which is a clear funnel. After logging both sides, `resid_sd_ratio_log = 1.0396`, so the spread is nearly the same in both halves and the funnel is gone.

**R1.5** Predictors with |coefficient| > 0.05 (constant excluded): ln(X4) `avg_bill_inr_thousands` (1.1576), X2 `missed_visit_count_12m` (0.0940), X3 `copay_burden_ratio` (0.1067) and X6 `chronic_condition_flag` (0.1518). This is only the mechanical 0.05 rule, since the coefficients are in different units. `mr_adj_r2 = 0.7403` against `loglog_adj_r2 = 0.6561`; adjusted R² already penalises the six extra columns and it still goes up by about 0.08, so the extra predictors do earn their place.

## Part 2 findings

Split check: 560 training rows (119 no-shows) and 240 test rows (51 no-shows).

**R2.1** After 200 updates: η = 1.0 gives 0.3703, η = 0.1 gives 0.3776, η = 0.01 gives 0.5091 (lowest to highest). A learning rate that is too small barely moves in 200 steps, while the large one got close to the minimum quickly; on this well-scaled problem η = 1.0 did not overshoot, though on a less smooth loss it could.

**R2.2** Largest absolute weights: X3 `copay_burden_ratio` (0.9013), X1 `months_registered` (-0.5554) and X2 `missed_visit_count_12m` (0.5505). The largest odds ratio is `odds_ratio_X3 = 2.4629`: a patient whose out-of-pocket share is one training SD higher has about 2.5 times the odds of missing the appointment, other things equal.

**R2.3** `blind_accuracy = 0.7875` against `tau_0.5_accuracy = 0.8042`. Saying "everyone shows up" is already 78.75% accurate while catching zero no-shows, so accuracy hides the thing we care about. Moving τ from 0.5 to 0.3 raised recall from 0.4706 to 0.7451 (24 to 38 no-shows caught out of 51) and lowered precision from 0.5455 to 0.4634 (false alarms went from 20 to 44).

**R2.4** Cross-validation picked K = 5 (`mean_cv_f1 = 0.4541`). On the test rows, logistic at τ = 0.3 has f1 0.5714 and recall 0.7451; KNN has f1 0.3696 and recall 0.3333. Logistic is clearly better on both, and KNN's majority vote at K = 5 acts like a fairly high threshold, so it misses most no-shows.

**R2.5** I would deploy the logistic model at τ = 0.15 (`cost_tau`, chosen on the training rows). On the test rows it costs INR 6,050 (`cost_test_at_tau`), against INR 11,800 at τ = 0.5 and INR 20,400 for flagging nobody. A missed no-show costs eight times a reminder call, so a low threshold that calls many patients is the cheaper policy. The threshold was picked on training rows so the test cost stays an honest estimate; choosing it on the test rows would tune to the same 240 rows we then report on.

## Part 3 findings

**R3.1** WCSS drops: K 2→3: 659.8282, 3→4: 108.0739, 4→5: 171.0575, 5→6: 45.5466. The elbow is at K = 3 (`elbow_k = 3`): the first drop is huge and the next one is less than a fifth of it. The drops do not shrink steadily (4→5 is bigger than 3→4); that comes from the pinned starting rows, which give one reproducible solution and not necessarily the best possible WCSS for every K.

**R3.2** Segments from `part3_segments.csv` (K = 3, ordered by copay burden):

| Segment | Name | Size | Profile | Action |
|---|---|---|---|---|
| A | Loyal regulars | 322 | ~47.5 months registered, ~2 missed visits, copay share 0.23, adherence 62, care-plan cost 3.60 | Low-touch: SMS confirmation only, no calls; offer self-service rebooking |
| B | Frequent reschedulers with big bills | 255 | ~28.6 months, ~6.1 missed visits, average bill 28.7k, care-plan cost 17.12 | Flexible slot booking and a confirmation step the day before; these drive most of the care-plan spend |
| C | New, cost-stretched patients | 223 | ~14.4 months, copay share 0.71, adherence 38.5 | Reminder call before every slot and a clear explanation of payment options |

Segment C has the highest `outcome_rate`, 0.4664: almost half of these patients missed their slot, against 0.0186 in segment A.

**R3.3** At the largest gap, single linkage gives 4 clusters: {E2, E3, E6}, {E1}, {E4}, {E5}. Complete linkage gives 2 clusters: {E1, E4, E5} and {E2, E3, E6}. Single linkage joins two groups through their closest pair, so its last merge happens at 3.4 (E5 to E6) and the biggest jump is earlier, between 1.4 and 2.8. Complete linkage uses the farthest pair, so the final merge is pushed up to 7.9 (E1 to E2) and that jump dominates, leaving two clear groups.

**R3.4** Step (1) of the PCA: after standardising, the mean vector is all zeros, so centring changes nothing. PC1 loads positively on X1 `months_registered` (0.5317) and X5 `adherence_score` (0.4418), and negatively on X3 `copay_burden_ratio` (-0.5187) and X2 `missed_visit_count_12m` (-0.4100), with X4 weaker (-0.2916). So PC1 is roughly "settled, adherent patient" versus "new, cost-stretched, often missing". It explains 52.27% of the variance alone, and 3 components are needed to reach 80% (`cumulative_ratio_3 = 0.8616`). Mean PC1 scores are 1.7773 (A), -0.7831 (B) and -1.6709 (C): the segments line up along PC1 in the same order as their no-show rates, so one axis already carries most of the segmentation.

## Part 4 findings

**R4.1** S4 (`months_registered <= 24`) is the best split by both information gain (0.1252) and Gini gain (0.0606): its yes side has 86 no-shows in 188 rows (45.7%) and its no side only 33 in 372 (8.9%), so it separates the classes more than any other candidate. A tree grown until every leaf is pure ends up with leaves holding one or two training patients, so it memorises noise in the 560 rows and does worse on new patients. A maximum depth (or a minimum number of rows per leaf) stops that.

**R4.2** I would ship the plain logistic model: f1 0.5714 and recall 0.7451 on the test rows. Combining did not help: hard vote got f1 0.5344 and recall 0.6863, soft vote f1 0.5507 and recall 0.7451 (same recall, more false alarms). The stump alone has the highest recall (0.7647) but lower f1 (0.5532), and the weak KNN (recall 0.3333) pulls the votes down.

**R4.3** After round 1, points 3 and 7 (the ones h1 got wrong) each have weight 0.25, so together they carry 0.5 of the total weight. In round 2 h2 has a weighted error of only 0.1667 against 0.25 for h1, so it gets a larger say: α2 = 0.8047 versus α1 = 0.5493.

**R4.4** Model A, option 2 (input 28 × 28 × 1):

| Layer | Output shape | Parameters |
|---|---|---|
| Conv 5×5, 6 filters, pad 2 | 28 × 28 × 6 | 156 |
| Max-pool 2×2 | 14 × 14 × 6 | 0 |
| Conv 5×5, 16 filters, pad 0 | 10 × 10 × 16 | 2,416 |
| Max-pool 2×2 | 5 × 5 × 16 | 0 |
| Flatten | 400 | 0 |
| Dense 32 | 32 | 12,832 |
| Dense 10 | 10 | 330 |

Total 15,734 (`cnn_total`), close to the 13,002 of the small MLP but with far more useful structure. The scaling example shows why: a dense layer on a 128 × 128 × 3 input with 1,024 neurons needs 50,332,672 parameters, while 64 3×3 filters need only 1,792, because filters are shared across positions.

Model B: W_xh is 128 × 64, W_hh is 128 × 128 and W_hy is 16 × 128 (plus b_h of 128 and b_y of 16), giving `rnn_total = 26768`. The same weights are reused at every time step, so a longer note means more steps, not more parameters.

For the 10-class model I would use ReLU in the hidden layers and a softmax output trained with categorical cross-entropy. In a very deep plain network the gradient is a product of many per-layer factors, so it can shrink towards zero by the time it reaches the first layers and they stop learning. A skip connection adds the layer's input to its output, giving the gradient a direct identity path back so early layers keep getting a useful signal.

## Model-choice memo

To: Head of Scheduling Operations
Re: Which no-show model to put into production

I recommend the logistic-regression model, with a reminder call to every patient whose predicted no-show probability is 0.15 or higher.

On the 240 held-out appointments, this threshold would have cost INR 6,050 in reminder calls and empty slots combined. The same model at the usual 0.5 cut-off would have cost INR 11,800, and doing nothing (no reminders) would have cost INR 20,400. The threshold was chosen on the 560 training appointments only, using INR 50 per unnecessary call and INR 400 per empty slot, so the test figure is a fair preview.

I compared it with four alternatives. At a 0.3 cut-off the logistic model caught 74.5% of no-shows (recall 0.7451, f1 0.5714). The nearest-neighbour model caught only 33.3%, and neither the hard vote (f1 0.5344) nor the soft vote (f1 0.5507) beat logistic on its own. Logistic is also the easiest to explain: a higher out-of-pocket share, more past missed visits and a short time on the platform all raise the risk.

Main risk: these numbers come from synthetic data, one 560/240 split and only 51 no-shows in the test set, so the INR figures could move noticeably on real appointments. I would run it on a pilot region and track actual call and empty-slot costs for a few weeks before rolling it out.

Neither Model A (scan routing) nor Model B (referral-note sorting) is worth building before this structured model is live and its savings are confirmed.

## Limitations

1. The data is synthetic. It was produced by a known generator with three clean hidden groups and a logistic outcome, so real patient data will be messier and the models will almost certainly score lower.
2. Every Part 2 and Part 4 result rests on a single fixed 70/30 split. With only 51 no-shows in the test set, a handful of patients changing sides would visibly shift recall, f1 and the INR costs.
3. The cost rule uses two flat figures (INR 50 per call, INR 400 per empty slot). Real costs vary by service, time slot and whether the slot can be re-filled, and `cost_tau` is sensitive to that ratio.
4. K-means uses pinned starting rows and stops at the first stable assignment, so the segments are one reproducible local solution, not necessarily the best one; the uneven WCSS drops show this.
5. The care-plan cost model is linear in logs and ignores interactions; the 95% interval in Part 1 assumes independent records and a normal sampling distribution.
