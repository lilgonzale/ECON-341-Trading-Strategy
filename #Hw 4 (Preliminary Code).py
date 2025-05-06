#Hw 4 (Preliminary Code)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- Parameters ---
np.random.seed(42)
initial_cash = 1_000_000
capital_fraction = 0.10
initial_price = 100
mu = 0.18
sigma_base = 0.10
steps_per_year = 252
T = 1
dt = 1 / steps_per_year
n_steps = int(T * steps_per_year)

# Option settings
option_horizon_days = 10
T_option = option_horizon_days / steps_per_year
rebalance_interval = option_horizon_days

# --- Black-Scholes ---
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# --- Simulate bond price path ---
prices = [initial_price]
volatility_history = [sigma_base]
for t in range(n_steps):
    sigma_t = sigma_base + 0.05 * np.sin(2 * np.pi * t / 126)
    dW = np.random.normal(0, np.sqrt(dt))
    new_price = prices[-1] * np.exp((mu - 0.5 * sigma_t ** 2) * dt + sigma_t * dW)
    prices.append(new_price)
    volatility_history.append(sigma_t)

rebalance_dates = range(rebalance_interval, len(prices), rebalance_interval)

# --- Expected straddle payoff from lognormal distribution ---
def expected_straddle_payoff(S, K, sigma, T):
    x = np.linspace(0.5 * S, 1.5 * S, 500)
    log_std = sigma * np.sqrt(T)
    pdf = norm.pdf((np.log(x / S)) / log_std) / (x * log_std)
    payoff = np.maximum(x - K, 0) + np.maximum(K - x, 0)
    return np.trapz(payoff * pdf, x)

# --- Strategy Execution ---
cash = initial_cash
portfolio_value = [initial_cash]

print("Rebalancing Log:\n------------------")
for i in rebalance_dates:
    S = prices[i]
    sigma = volatility_history[i]
    r = 0.00

    strike_range = np.round(np.linspace(0.95 * S, 1.05 * S, 11), 2)

    best_net_profit = -np.inf
    best_K = None
    best_straddle_cost = None

    for K in strike_range:
        expected_payoff = expected_straddle_payoff(S, K, sigma, T_option)
        call_premium = black_scholes_call(S, K, T_option, r, sigma)
        put_premium = black_scholes_put(S, K, T_option, r, sigma)
        premium = call_premium + put_premium
        net_profit = expected_payoff - premium

        if net_profit > best_net_profit:
            best_net_profit = net_profit
            best_K = K
            best_straddle_cost = premium

    deployable_capital = cash * capital_fraction
    num_contracts = deployable_capital // (best_straddle_cost * 100)

    print(f"Day {i} | Price: {S:.2f} | Vol: {sigma:.3f} | Best Strike: {best_K:.2f} | Net Exp. Profit: {best_net_profit:.2f}")

    if num_contracts < 1:
        print("  → Not enough capital to buy even 1 contract.\n")
        portfolio_value.append(cash)
        continue

    invested_amount = num_contracts * best_straddle_cost * 100
    S_future = prices[min(i + option_horizon_days, len(prices) - 1)]
    payoff = (max(S_future - best_K, 0) + max(best_K - S_future, 0)) * 100 * num_contracts

    cash = cash - invested_amount + payoff
    portfolio_value.append(cash)

    print(f"  → Contracts: {int(num_contracts)} | Future Price: {S_future:.2f} | Payoff: ${payoff:,.2f} | Cash: ${cash:,.2f}\n")

# --- Final Results ---
final_value = cash
profit = final_value - initial_cash
return_dict = {
    "Final Portfolio Value": final_value,
    "Total Profit/Loss": profit,
    "Return (%)": (profit / initial_cash) * 100
}

print("Final Portfolio Value: ${:,.2f}".format(final_value))
print("Total Profit/Loss: ${:,.2f}".format(profit))
print("Return: {:.2f}%".format(return_dict['Return (%)']))

# --- Plot Portfolio Value ---
plt.figure(figsize=(10, 5))
plt.plot(np.linspace(0, T, len(portfolio_value)), portfolio_value, label='Portfolio Value')
plt.title("Always-In Strategy: Best Strike (Net Expected Profit, 95–105% Band)")
plt.xlabel("Time (Years)")
plt.ylabel("Portfolio Value ($)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()