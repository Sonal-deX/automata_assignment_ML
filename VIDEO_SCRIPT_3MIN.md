# 3-Minute Video Presentation Script (Story / Promo Style)

**Total Duration:** ~2 minutes 45 seconds (fits strictly under the 3-minute limit)  
**What to show on screen:**
- **0:00 – 0:35:** Dashboard `Overview` tab (showing the 12 Sri Lankan cities table & monsoon rainfall chart)
- **0:35 – 1:45:** Jupyter Notebook `ensemble_rainfall_sri_lanka.ipynb` (scrolling through the Bagging curve, Boosting walk-forward curve, ARMA+RF & Stacking weights, and comparison bar chart)
- **1:45 – 2:45:** Dashboard `Model comparison`, `Forecast a past day` (e.g., Ratnapura or Colombo in May 2024), and `Try your own values` live slider demo.

---

### [0:00 – 0:35] Hook & The Sri Lankan Problem (35 sec)
> "In Sri Lanka, one question affects almost everything—from paddy farmers planning Yala and Maha cultivation, to tea estates in Nuwara Eliya, to flood warnings along the Kelani and Kalu rivers in Colombo and Ratnapura: **given today's weather, how much rain will fall tomorrow?**
>
> What makes Sri Lanka unique is that rainfall isn't uniform. During the South-West monsoon from May to September, Ratnapura and Galle are drenched while Trincomalee and Jaffna stay dry in the rain shadow of the central hills—and from November to January, the North-East monsoon flips that pattern completely.
>
> To model this, I collected **10 years of real daily meteorological data from 2015 to 2024 across 12 Sri Lankan cities** covering the Wet, Intermediate, and Dry zones—over **43,800 daily observations**."

---

### [0:35 – 1:45] Why Ensembles & Hybrids? What We Discovered (70 sec)
> "A single decision tree completely fails on daily rainfall—it memorizes past extreme storms and gets a negative R-squared of `-0.14` on our unseen 2023–2024 test years.
>
> Here is how different ensemble paradigms behave on this dataset and *why*:
> 1. **Bagging**—Random Forest and Extra Trees—trains deep trees in parallel and averages them. As our variance-reduction curve shows, going from 1 tree to 200 trees cancels out the variance from noisy tropical rainstorms and drops test RMSE from `9.44 mm` down to `8.15 mm` without ever overfitting.
> 2. **Boosting**—XGBoost, LightGBM, and CatBoost—trains shallow trees sequentially on residuals. Because daily rainfall has random convective spikes, boosting actually *overfits* if left running for 600+ rounds. By using 3-year walk-forward temporal validation, we stopped boosting at the sweet spot around 80 to 166 rounds. **CatBoost** led the boosting models because its symmetric trees act as a strong regularizer and it handles Sri Lankan city, zone, and monsoon categories natively.
> 3. **Time-Series Hybridization (`ARMA + Random Forest`)**: A pure `ARMA(2,1)` model captures multi-day monsoon persistence (`8.45 mm` RMSE), but it only sees past rain numbers—so it's blind when barometric pressure suddenly drops or winds shift. When we trained a Random Forest on the **ARMA residuals** using humidity, wind direction, radiation, and pressure change, RMSE dropped to `8.15 mm`.
> 4. **Tree + Neural Network Hybrid & Stacking**: Decision trees make step-function splits, while our Multi-Layer Perceptron neural network learns a smooth continuous surface. When our **Stacking Ensemble** combined 6 base models—Random Forest, Extra Trees, XGBoost, LightGBM, CatBoost, and the ANN—using a non-negative Ridge meta-learner, it assigned weight across *all three families* (`31%` ANN, `51%` Boosting, `28%` Bagging) and achieved the **lowest overall RMSE (`8.08 mm`), highest R-squared (`0.394`), and best Rain-Day F1 score (`0.842`)** across every climatic zone in Sri Lanka."

---

### [1:45 – 2:45] Live Dashboard Demo & Wrap-Up (60 sec)
> "To make these models practical, I built a clean Streamlit dashboard backed by our saved pickle models.
>
> - On the **Model comparison** tab, we can inspect all 17 models, compare RMSE across Wet, Intermediate, and Dry zones, and see how permutation importance ranks today's rain hours, monsoon wind direction, solar radiation, and 24-hour pressure change as the top physical drivers.
> - On **Forecast a past day**, we can pick any real day from 2023–2024—like a South-West monsoon day in Colombo or Ratnapura—and compare every model's next-day forecast against what actually fell.
> - Finally, on **Try your own values**, we can adjust sliders for today's rainfall, humidity, cloud cover, and pressure and watch all 9 live pickle models update their tomorrow forecast in real time.
>
> All code, the notebook, trained pickle models, and our AI usage declaration are available in the GitHub repository linked in my submission. Thank you!"
