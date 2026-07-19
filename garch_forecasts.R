# ============================================================
# GARCH baselines for the validation-regime study.
# GARCH-Normal and GJR-GARCH-t, each as a static fit (frozen on 2000-2012)
# and a rolling expanding-window refit. Exports the four forecast series the
# Python combined panel reads, plus the realized variance for alignment.
#
# Input: sp500_for_garch.csv, with columns date, returns, rv_gk, one row per
# trading day. The main notebook (validation_regime_experiment.ipynb) writes
# this file, so the GARCH forecasts use the same S&P 500 series as the DL models.
# ============================================================
library(rugarch)

# ============================================================
# 1. LOAD DATA
# ============================================================
df <- read.csv("sp500_for_garch.csv")
df$date <- as.Date(df$date)

# Split boundaries (match the DL pipeline).
# Train: 2000 .. 2012-12-31. Test: 2020-01-01 onward (frozen).
# GARCH has no validation hold-out; the 2013-16 validation windows are a DL concept.
train <- df[df$date <  as.Date("2013-01-01"), ]
test  <- df[df$date >= as.Date("2020-01-01"), ]

returns_train <- train$returns
returns_test  <- test$returns
rv_actual     <- test$rv_gk          # test target (variance), for sanity and alignment
n_test  <- nrow(test)
n_train <- length(returns_train)

# returns_all is pre-2013 fit-data glued to 2020+ test; 2013-2019 is intentionally
# excluded so GARCH is span-matched to the DL training window (no recency advantage).
returns_all <- c(returns_train, returns_test)
cat("Train (2000-2012):", n_train, "| Test (2020+):", n_test, "\n")

# ============================================================
# 2. SPECIFICATIONS
# ============================================================
spec_norm <- ugarchspec(
  variance.model     = list(model = "sGARCH", garchOrder = c(1, 1)),
  mean.model         = list(armaOrder = c(0, 0), include.mean = TRUE),
  distribution.model = "norm"
)
spec_gjr <- ugarchspec(
  variance.model     = list(model = "gjrGARCH", garchOrder = c(1, 1)),
  mean.model         = list(armaOrder = c(0, 0), include.mean = TRUE),
  distribution.model = "std"
)

# ============================================================
# 3. STATIC forecasts: fit ONCE on 2000-2012, forecast n_test ahead.
#    Frozen params go stale by 2020. This is the static side of the contrast.
# ============================================================
static_forecast <- function(spec, returns_train, n_test, name) {
  cat("Static fit:", name, "\n")
  fit <- ugarchfit(spec, returns_train, solver = "hybrid")
  fc  <- ugarchforecast(fit, n.ahead = n_test)
  as.numeric(sigma(fc))^2     # variance, same scale as rv_gk
}

f_norm_static <- static_forecast(spec_norm, returns_train, n_test, "GARCH-Normal")
f_gjr_static  <- static_forecast(spec_gjr,  returns_train, n_test, "GJR-GARCH-t")

# ============================================================
# 4. ROLLING forecasts: expanding window from the train boundary, daily refit.
#    Span-matched: returns_all is c(train, test), so 2013-2019 is excluded from
#    estimation and the window walks pre-2013 into the 2020+ test only.
# ============================================================
rolling_forecast <- function(spec, returns_all, n_train, n_test, name) {
  cat("Forecasting:", name, "\n")
  forecasts <- numeric(n_test)
  for (i in 1:n_test) {
    window <- returns_all[1:(n_train + i - 1)]
    fit <- tryCatch(ugarchfit(spec, window, solver = "hybrid"), error = function(e) NULL)
    if (is.null(fit)) {
      forecasts[i] <- ifelse(i > 1, forecasts[i - 1], var(window))
    } else {
      forecasts[i] <- as.numeric(sigma(ugarchforecast(fit, n.ahead = 1)))^2
    }
    if (i %% 200 == 0) cat("  ", i, "/", n_test, "\n")
  }
  cat("Done\n")
  forecasts
}

f_norm_roll <- rolling_forecast(spec_norm, returns_all, n_train, n_test, "GARCH-Normal rolling")
f_gjr_roll  <- rolling_forecast(spec_gjr,  returns_all, n_train, n_test, "GJR-GARCH-t rolling")

# ============================================================
# 5. SANITY + EXPORT
# ============================================================
cat("\n=== Sanity (test-set means; rv_actual mean =", round(mean(rv_actual), 8), ") ===\n")
cat("GARCH-N static  mean:", round(mean(f_norm_static), 8), "\n")
cat("GJR-t   static  mean:", round(mean(f_gjr_static),  8), "\n")
cat("GARCH-N rolling mean:", round(mean(f_norm_roll),   8), "\n")
cat("GJR-t   rolling mean:", round(mean(f_gjr_roll),    8), "\n")

# One row per test day; columns are variance forecasts on the rv_gk scale.
# actual_rv_gk and returns_test are included so the Python eval can self-check alignment.
out <- data.frame(
  date              = test$date,
  actual_rv_gk      = rv_actual,
  returns_test      = returns_test,
  garch_norm_static = f_norm_static,
  gjr_t_static      = f_gjr_static,
  garch_norm_roll   = f_norm_roll,
  gjr_t_roll        = f_gjr_roll
)
write.csv(out, "garch_forecasts.csv", row.names = FALSE)
cat("\nSaved garch_forecasts.csv  (", nrow(out), "rows )\n")
